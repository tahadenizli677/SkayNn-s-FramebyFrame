# Arsenal YOLOv8 aimbot - pure onnxruntime, torch-free - Python 3.11+, Windows 11, PyInstaller target
# model: best.onnx next to the exe (train.py drops it there), config: aimbot.ini (auto-created)
# bare run = menu | --live = straight into aim loop | --selftest = one-frame pipeline check

import configparser
import ctypes
import ctypes.wintypes as wt
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path

if sys.stdout is None or sys.stderr is None:
    # --windowed build has no console: keep prints in aimbot.log next to the exe
    _base = (Path(sys.executable).resolve().parent
             if getattr(sys, "frozen", False) else Path(__file__).resolve().parent)
    _log = open(_base / "aimbot.log", "w", encoding="utf-8", buffering=1)
    sys.stdout = _log
    sys.stderr = _log

print("loading...", flush=True)   # heavy imports below take a few seconds

import cv2
import dxcam
import numpy as np
import onnxruntime as ort

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
VK_END = 0x23

STOP_EVENT = threading.Event()   # GUI stop button / END key both raise this

DEFAULTS = {
    "aim": {
        "key": "mouse4",        # XButton1 (side button), hold
        "toggle_key": "home",   # Home, arm/disarm
        "conf": "0.35",
        "fov": "400",           # px radius in capture space
        "smooth": "0.45",       # 0 = snap, 1 = glide
        "gain": "1.0",          # mouse counts per px, match roblox sens
        "head_frac": "0.15",    # aim point measured from top of bbox
        "lead": "1",            # frames of latency compensation
        "classes": "0",         # comma-separated target class ids from data.yaml
    },
    "capture": {"region": "", "fps": "240"},   # region = l,t,r,b or empty for full screen
    "model": {"path": "best.onnx"},
    "detect": {
        "mode": "color",        # color = neon outline, yolo = best.onnx
        "low": "26,160,190",    # HSV low  for the bright enemy outline
        "high": "34,255,255",   # HSV high for the bright enemy outline
        "min_area": "250",      # ignore blobs smaller than this (px^2)
        "max_area": "250000",   # ignore blobs bigger than this
        "max_fill": "0.55",     # solid yellow props/ui are not hollow outlines
    },
    "debug": {"overlay": "True"},   # draw boxes + FOV circle on the game
}

# Friendly key names -> Windows virtual-key codes. The ini stores the name
# ("mouse4", "f6", "x") so the file reads like something a human typed;
# legacy "0x05" hex and plain decimals still parse.
KEY_ALIASES = {
    "mouse1": 0x01, "mouse2": 0x02, "mouse3": 0x04,      # left, right, middle
    "mouse4": 0x05, "mouse5": 0x06,                     # side buttons (X1/X2)
    "x1": 0x05, "x2": 0x06,
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "return": 0x0D,
    "shift": 0x10, "ctrl": 0x11, "alt": 0x12, "pause": 0x13,
    "caps": 0x14, "esc": 0x1B, "escape": 0x1B, "space": 0x20,
    "pgup": 0x21, "pgdn": 0x22, "pageup": 0x21, "pagedown": 0x22,
    "end": 0x23, "home": 0x24,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "insert": 0x2D, "delete": 0x2E, "numlock": 0x90, "scrolllock": 0x91,
}
KEY_CANON = {
    0x01: "mouse1", 0x02: "mouse2", 0x04: "mouse3", 0x05: "mouse4",
    0x06: "mouse5", 0x08: "backspace", 0x09: "tab", 0x0D: "enter",
    0x10: "shift", 0x11: "ctrl", 0x12: "alt", 0x13: "pause", 0x14: "caps",
    0x1B: "esc", 0x20: "space", 0x21: "pgup", 0x22: "pgdn", 0x23: "end",
    0x24: "home", 0x25: "left", 0x26: "up", 0x27: "right", 0x28: "down",
    0x2D: "insert", 0x2E: "delete", 0x90: "numlock", 0x91: "scrolllock",
}


def parse_key(text) -> int:
    """'mouse4' / 'f6' / 'x' / '0x05' / '69' -> virtual-key code."""
    s = str(text).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if not s:
        raise ValueError("empty key")
    if s in KEY_ALIASES:
        return KEY_ALIASES[s]
    if s.startswith("0x") or s.startswith("&h"):
        return int(s, 16)
    if s.startswith("f") and s[1:].isdigit() and 1 <= int(s[1:]) <= 24:
        return 0x6F + int(s[1:])       # VK_F1 = 0x70
    if len(s) == 1:
        return ord(s.upper())          # a-z -> 0x41..0x5A, 0-9 -> 0x30..0x39
    if s.isdigit():
        return int(s)                  # raw vk number from old configs
    raise ValueError(f"unknown key: {text!r}")


def key_name(vk: int) -> str:
    """Virtual-key code -> the readable name stored in the ini / log."""
    if vk in KEY_CANON:
        return KEY_CANON[vk]
    if 0x70 <= vk <= 0x87:
        return f"f{vk - 0x6F}"
    if 0x41 <= vk <= 0x5A or 0x30 <= vk <= 0x39:
        return chr(vk).lower()
    if 0x60 <= vk <= 0x69:
        return f"num{vk - 0x60}"
    return f"0x{vk:02x}"


def key_text(value) -> str:
    """ini field validator: any accepted spelling -> canonical key name."""
    return key_name(parse_key(value))


def key_show(value) -> str:
    """Display-only: never raises, falls back to the raw ini value."""
    try:
        return key_text(value)
    except ValueError:
        return str(value)


TUNE_FIELDS = [
    ("detect", "mode", str, "color or yolo"),
    ("detect", "low", str, "outline HSV low"),
    ("detect", "high", str, "outline HSV high"),
    ("detect", "min_area", int, "ignore smaller blobs"),
    ("aim", "conf", float, "detection confidence"),
    ("aim", "fov", float, "engagement radius px"),
    ("aim", "smooth", float, "0=snap 1=glide"),
    ("aim", "gain", float, "mouse counts per px"),
    ("aim", "head_frac", float, "aim point from bbox top"),
    ("aim", "lead", int, "latency frames"),
    ("aim", "classes", str, "target class ids (csv)"),
    ("aim", "key", key_text, "aim key: mouse4, f6, x, shift..."),
    ("aim", "toggle_key", key_text, "toggle key: home, f9, caps..."),
    ("capture", "region", str, "l,t,r,b or empty=fullscreen"),
    ("capture", "fps", int, "capture fps"),
    ("model", "path", str, "model file"),
    ("debug", "overlay", str, "True/False game overlay"),
]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wt.LONG),
        ("dy", wt.LONG),
        ("mouseData", wt.DWORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUT(ctypes.Structure):
    class _UNION(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT)]

    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _UNION)]


def move_mouse(dx: int, dy: int) -> None:
    # Send both mouse_event and SendInput for maximum compatibility with Roblox
    try:
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)
    except Exception:
        pass
    try:
        inp = _INPUT(type=INPUT_MOUSE)
        inp.mi.dx, inp.mi.dy, inp.mi.dwFlags = int(dx), int(dy), MOUSEEVENTF_MOVE
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    except Exception:
        pass


def key_down(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundle_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return app_dir()


def ini_path() -> Path:
    return app_dir() / "aimbot.ini"


def load_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    for section, values in DEFAULTS.items():
        cfg[section] = dict(values)
    path = ini_path()
    if path.exists():
        cfg.read(path, encoding="utf-8")
    else:
        bundled = bundle_dir() / "aimbot.ini"
        if bundled.exists() and bundled != path:
            try:
                cfg.read(bundled, encoding="utf-8")
            except Exception:
                pass
        with open(path, "w", encoding="utf-8") as f:
            cfg.write(f)
    return cfg


def save_config(cfg: configparser.ConfigParser) -> None:
    with open(ini_path(), "w", encoding="utf-8") as f:
        cfg.write(f)


def letterbox(frame: np.ndarray, size: int = 640):
    h, w = frame.shape[:2]
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, r, dx, dy


def decode(raw: np.ndarray) -> np.ndarray:
    """yolov8 export layout: (1, 4+nc, 8400), cxcywh in 640 input px, class logits or probs.
    range guards also accept transposed (N, 4+nc) and normalized variants."""
    out = raw[0]
    if out.ndim == 2 and out.shape[0] < out.shape[1]:
        out = out.T
    boxes, scores = out[:, :4].copy(), out[:, 4:]
    if scores.size and (scores.max() > 1.0 + 1e-6 or scores.min() < -1e-6):
        scores = 1.0 / (1.0 + np.exp(-scores))   # raw logits -> sigmoid
    if boxes.size and boxes.max() <= 1.0 + 1e-6:
        boxes *= 640.0                           # normalized coordinates -> px
    x1, y1 = boxes[:, 0] - boxes[:, 2] * 0.5, boxes[:, 1] - boxes[:, 3] * 0.5
    x2, y2 = boxes[:, 0] + boxes[:, 2] * 0.5, boxes[:, 1] + boxes[:, 3] * 0.5
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)     # (N, 4) in 640 space
    return np.concatenate([boxes_xyxy, scores], axis=1)  # (N, 4+nc)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    bb = np.atleast_2d(np.asarray(b, dtype=float))
    ix1 = np.maximum(a[0], bb[:, 0])
    iy1 = np.maximum(a[1], bb[:, 1])
    ix2 = np.minimum(a[2], bb[:, 2])
    iy2 = np.minimum(a[3], bb[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def parse_hsv(text) -> tuple:
    """'H,S,V' -> clamped OpenCV triple (H is 0..179, S/V 0..255)."""
    try:
        parts = [int(v) for v in str(text).split(",")]
    except ValueError:
        parts = [0, 0, 0]
    while len(parts) < 3:
        parts.append(0)
    h, s, v = parts[:3]
    if h > 179:                     # hand-typed values often use 0..255 hue
        h = round(h * 179 / 255)
    return (max(0, min(179, h)),
            max(0, min(255, s)),
            max(0, min(255, v)))


def hsv_bounds(low_text, high_text) -> tuple:
    """Clamp both bounds and un-invert them.

    An inverted channel (low > high) would empty cv2.inRange and silently
    kill detection, so swap any channel that came out backwards.
    """
    low, high = list(parse_hsv(low_text)), list(parse_hsv(high_text))
    for i in range(3):
        if low[i] > high[i]:
            low[i], high[i] = high[i], low[i]
    return tuple(low), tuple(high)


def color_rows(frame: np.ndarray, low, high, min_area: int, max_area: int,
               max_fill: float = 0.55) -> np.ndarray:
    """Enemy outline detector: HSV threshold -> contours -> player boxes.

    Arsenal draws a thin neon outline around enemies, so this needs no
    training data and no GPU - a full 1280x960 frame costs a couple of ms.
    Map props often share the hue, so a blob only counts as a player when it
    is upright, mostly empty inside (thin ring, not a painted rectangle) and
    actually encloses background (a real hole in the contour).
    Output matches decode(): (x1, y1, x2, y2, conf, cls).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.asarray(low, np.uint8), np.asarray(high, np.uint8))
    if not mask.any():
        return np.empty((0, 6), np.float32)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return np.empty((0, 6), np.float32)
    hierarchy = hierarchy[0]
    rows = []
    effective_min_area = min(min_area, 35)
    for i, c in enumerate(contours):
        if hierarchy[i][3] != -1:               # this contour is a hole
            continue
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if not (effective_min_area <= area <= max_area):
            continue
        fill = cv2.countNonZero(mask[y:y + h, x:x + w]) / float(area)
        is_small = area < 300
        if is_small:
            # Distant targets: thin outline closes in, fill is higher and no background hole
            if fill > 0.88:
                continue
            if h < w * 0.60 or h > w * 6.0:
                continue
        else:
            # Close/medium targets: must be upright hollow outline
            if fill > max_fill:
                continue
            if hierarchy[i][2] == -1 and fill > 0.30:
                continue
            if h < w * 0.85 or h > w * 5.0:
                continue
        rows.append((x, y, x + w, y + h, 1.0, 0.0))
    if not rows:
        return np.empty((0, 6), np.float32)
    return np.asarray(rows, np.float32)


def sample_color(cfg: configparser.ConfigParser, seconds: float = 3.0) -> bool:
    """Auto-fit the outline HSV bounds from whatever is on screen right now.

    Point the game at an enemy (outline clearly visible) and press the GUI
    button: grabs frames for `seconds`, takes the biggest upright hollow
    yellow-ish blob, and writes p5..p95 of its pixels as the new bounds.
    Returns True when bounds were found.
    """
    camera = create_camera(cfg)
    low_all, high_all = None, None
    best_n, t_end = 0, time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # deliberately broad: we only want "bright yellow-ish" candidates
        mask = cv2.inRange(hsv, np.array((10, 90, 110), np.uint8),
                           np.array((50, 255, 255), np.uint8))
        if not mask.any():
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if not (250 <= area <= 250000) or h < w * 0.9 or h > w * 5.0:
                continue
            if cv2.countNonZero(mask[y:y + h, x:x + w]) / float(area) > 0.6:
                continue                      # solid yellow prop, not outline
            patch, pm = hsv[y:y + h, x:x + w], mask[y:y + h, x:x + w] > 0
            if pm.sum() < best_n:
                continue
            hs = patch[:, :, 0][pm].astype(int)
            ss = patch[:, :, 1][pm].astype(int)
            vs = patch[:, :, 2][pm].astype(int)
            best_n = int(pm.sum())
            low_all = (int(np.percentile(hs, 5)) - 4,
                       int(np.percentile(ss, 15)),
                       int(np.percentile(vs, 15)))
            high_all = (int(np.percentile(hs, 95)) + 4,
                        int(np.percentile(ss, 95)) + 20,
                        255)
        time.sleep(0.05)
    camera.stop()
    if low_all is None:
        return False
    clamp = lambda v: max(0, min(255, v))
    low = tuple(clamp(v) for v in low_all)
    high = tuple(max(h, l + 1) for h, l in zip(
        (clamp(v) for v in high_all), low))
    cfg.set("detect", "low", ",".join(str(v) for v in low))
    cfg.set("detect", "high", ",".join(str(v) for v in high))
    save_config(cfg)
    print(f"sampled outline: hsv {low} -> {high} ({best_n} px)")
    return True


def select_target(rows: np.ndarray, center, fov: float, allowed: set[int],
                  conf_thr: float, head_frac: float, prev):
    """rows = (x1,y1,x2,y2,scores...) in capture px; nearest-to-crosshair with fov gate"""
    cx, cy = center
    if not rows.size:
        return None, np.empty((0, 4)), None
    scores = rows[:, 4:]
    cls_ids = scores.argmax(axis=1)
    confs = scores[np.arange(scores.shape[0]), cls_ids]
    mask = (confs >= conf_thr) & np.isin(cls_ids, list(allowed))
    boxes = rows[mask][:, :4]
    if not boxes.size:
        return None, np.empty((0, 4)), None

    # cheap duplicate suppression: same enemy twice -> taller box wins (head is stable)
    confs_masked = confs[mask]
    heights = boxes[:, 3] - boxes[:, 1]
    order = np.lexsort((-heights, -confs_masked))
    boxes = boxes[order]
    kept = []
    for i in range(boxes.shape[0]):
        if all(iou(boxes[i], boxes[j]).max(initial=0.0) <= 0.5 for j in kept):
            kept.append(i)
        if len(kept) >= 16:
            break
    boxes = boxes[kept]

    ax = (boxes[:, 0] + boxes[:, 2]) * 0.5
    ay = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) * head_frac
    dist = np.hypot(ax - cx, ay - cy)
    in_fov = dist <= fov
    if not in_fov.any():
        return None, boxes, None
    best = int(np.argmin(np.where(in_fov, dist, np.inf)))
    target = (float(ax[best]), float(ay[best]))
    if prev and (target[0] - prev[0]) ** 2 + (target[1] - prev[1]) ** 2 > 200 ** 2:
        prev = None   # unit swap, do not continue the velocity line
    return target, boxes, prev


def lead_pos(target, prev, frames: int):
    if not prev or frames <= 0:
        return target
    return (target[0] + (target[0] - prev[0]) * frames,
            target[1] + (target[1] - prev[1]) * frames)


def open_session(path: Path) -> tuple[ort.InferenceSession, str]:
    if not path.exists():
        alt = app_dir() / "best.onnx"
        if alt != path and alt.exists():
            path = alt
        else:
            bundled = bundle_dir() / "best.onnx"
            if bundled.exists():
                path = bundled
    if not path.exists():
        sys.exit(f"model not found: {path} -> run train.py")
    # provider list is already ordered by preference, cuda first with onnxruntime-gpu
    session = ort.InferenceSession(str(path), providers=ort.get_available_providers())
    return session, session.get_inputs()[0].name


def create_camera(cfg: configparser.ConfigParser):
    region_raw = cfg["capture"]["region"].strip()
    region = tuple(int(v) for v in region_raw.split(",")) if region_raw else None
    camera = dxcam.create(output_idx=0, region=region)
    if camera is None:
        sys.exit("dxcam: no dxgi adapter")
    # dxcam 0.3 api: fps lives on start(), video_mode drains the ring buffer to latest frame
    camera.start(target_fps=cfg["capture"].getint("fps"), video_mode=True)
    return camera


def selftest(cfg: configparser.ConfigParser) -> None:
    """one frame end-to-end: capture -> detect, no mouse input"""
    mode = cfg["detect"].get("mode", "color").strip().lower()
    session = input_name = None
    if mode == "yolo":
        session, input_name = open_session(app_dir() / cfg["model"]["path"])
    camera = create_camera(cfg)
    if session:
        print(f"providers={session.get_providers()}")

    frame, deadline = None, time.perf_counter() + 3.0
    while frame is None and time.perf_counter() < deadline:
        frame = camera.get_latest_frame()
        time.sleep(0.01)
    if frame is None:
        camera.stop()
        sys.exit("selftest: dxcam produced no frames")

    if mode == "color":
        low, high = hsv_bounds(cfg["detect"]["low"], cfg["detect"]["high"])
        min_area = cfg["detect"].getint("min_area")
        max_area = cfg["detect"].getint("max_area")
        max_fill = cfg["detect"].getfloat("max_fill")
        t0 = time.perf_counter()
        rows = color_rows(frame, low, high, min_area, max_area, max_fill)
        ms = (time.perf_counter() - t0) * 1000
        print(f"selftest ok  frame={frame.shape[1]}x{frame.shape[0]}  "
              f"detect=color  boxes={len(rows)}")
        print(f"color={ms:.2f}ms  hsv={low}->{high}  area={min_area}..{max_area}")
        for row in rows[:8]:
            x1, y1, x2, y2 = row[:4].astype(int)
            print(f"  box {x1},{y1} -> {x2},{y2}  ({x2 - x1}x{y2 - y1})")
        camera.stop()
        return

    canvas, r, dx, dy = letterbox(frame)
    blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    t0 = time.perf_counter()
    raw = session.run(None, {input_name: blob})[0]
    ms = (time.perf_counter() - t0) * 1000
    rows = decode(raw)
    scores = rows[:, 4:]

    print(f"selftest ok  frame={frame.shape[1]}x{frame.shape[0]}  raw={raw.shape}  decoded={rows.shape}")
    print(f"infer={ms:.1f}ms  providers={session.get_providers()}")
    if scores.size:
        print(f"score range [{scores.min():.3f}, {scores.max():.3f}]  box_max={rows[:, :4].max():.1f} (640-space)")
    camera.stop()


_USER32 = ctypes.windll.user32
_GDI32 = ctypes.windll.gdi32
_KERNEL32 = ctypes.windll.kernel32

_WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wt.HWND, wt.UINT,
                              wt.WPARAM, wt.LPARAM)
_overlay_wndproc = None


class _POINT(ctypes.Structure):
    _fields_ = [("x", wt.LONG), ("y", wt.LONG)]


class _SIZE(ctypes.Structure):
    _fields_ = [("cx", wt.LONG), ("cy", wt.LONG)]


class _BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte),
                ("BlendFlags", ctypes.c_ubyte),
                ("SourceConstantAlpha", ctypes.c_ubyte),
                ("AlphaFormat", ctypes.c_ubyte)]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG),
                ("biHeight", wt.LONG), ("biPlanes", wt.WORD),
                ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", wt.LONG),
                ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                ("biClrImportant", wt.DWORD)]


class _WNDCLASSEXW(ctypes.Structure):
    _fields_ = [("cbSize", wt.UINT), ("style", wt.UINT),
                ("lpfnWndProc", ctypes.c_void_p), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wt.HINSTANCE),
                ("hIcon", wt.HICON), ("hCursor", wt.HANDLE),
                ("hbrBackground", wt.HBRUSH), ("lpszMenuName", wt.LPCWSTR),
                ("lpszClassName", wt.LPCWSTR), ("hIconSm", wt.HICON)]


def _register_overlay_class() -> None:
    """One window class for the whole process; keeps the WNDPROC alive."""
    global _overlay_wndproc
    if _overlay_wndproc is not None:
        return
    _USER32.DefWindowProcW.restype = ctypes.c_ssize_t
    _USER32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
    proc = _WNDPROC(lambda hwnd, msg, wp, lp:
                    _USER32.DefWindowProcW(hwnd, msg, wp, lp))
    wc = _WNDCLASSEXW()
    wc.cbSize = ctypes.sizeof(_WNDCLASSEXW)
    wc.lpfnWndProc = ctypes.cast(proc, ctypes.c_void_p)
    wc.hInstance = _KERNEL32.GetModuleHandleW(None)
    wc.lpszClassName = "ArsenalOverlay"
    _USER32.RegisterClassExW.argtypes = [ctypes.POINTER(_WNDCLASSEXW)]
    _USER32.RegisterClassExW.restype = wt.ATOM
    if not _USER32.RegisterClassExW(ctypes.byref(wc)):
        if ctypes.get_last_error() not in (0, 1410):   # 1410 = already exists
            raise ctypes.WinError()
    _overlay_wndproc = proc


class Overlay:
    """Boxes drawn straight on the game - no separate window, no frame copy.

    A layered popup the size of the captured area, painted through
    UpdateLayeredWindow with per-pixel alpha: everything not drawn stays fully
    transparent, so only the boxes/circle/crosshair are visible over the game.
    WS_EX_TRANSPARENT + WS_EX_NOACTIVATE keep it click-through and unfocusable,
    HWND_TOPMOST keeps it above borderless-fullscreen games, WS_EX_TOOLWINDOW
    hides it from the taskbar and alt-tab.
    """

    def __init__(self, x: int, y: int, w: int, h: int) -> None:
        _register_overlay_class()
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        ex_style = (0x00080000 |   # WS_EX_LAYERED
                    0x00000020 |   # WS_EX_TRANSPARENT (click-through)
                    0x00000008 |   # WS_EX_TOPMOST
                    0x00000080 |   # WS_EX_TOOLWINDOW
                    0x08000000)    # WS_EX_NOACTIVATE
        _USER32.CreateWindowExW.argtypes = [
            wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wt.HWND, wt.HMENU, wt.HINSTANCE, ctypes.c_void_p]
        _USER32.CreateWindowExW.restype = wt.HWND
        self.hwnd = _USER32.CreateWindowExW(
            ex_style, "ArsenalOverlay", "ArsenalOverlay", 0x80000000,  # WS_POPUP
            self.x, self.y, self.w, self.h, None, None,
            _KERNEL32.GetModuleHandleW(None), None)
        if not self.hwnd:
            raise ctypes.WinError()

        _USER32.SetWindowPos.argtypes = [wt.HWND, wt.HWND, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int, wt.UINT]
        _USER32.SetWindowPos.restype = wt.BOOL
        _USER32.SetWindowPos(self.hwnd, wt.HWND(-1), self.x, self.y,
                             self.w, self.h, 0x0040)   # SWP_SHOWWINDOW

        _USER32.GetDC.argtypes = [wt.HWND]
        _USER32.GetDC.restype = wt.HDC
        _GDI32.CreateCompatibleDC.argtypes = [wt.HDC]
        _GDI32.CreateCompatibleDC.restype = wt.HDC
        _GDI32.CreateDIBSection.argtypes = [
            wt.HDC, ctypes.c_void_p, wt.UINT,
            ctypes.POINTER(ctypes.c_void_p), wt.HANDLE, wt.DWORD]
        _GDI32.CreateDIBSection.restype = wt.HANDLE
        _GDI32.SelectObject.argtypes = [wt.HDC, wt.HANDLE]
        _GDI32.SelectObject.restype = wt.HANDLE
        _USER32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]

        screen_dc = _USER32.GetDC(None)
        self.mem_dc = _GDI32.CreateCompatibleDC(screen_dc)
        bmi = _BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.biWidth = self.w
        bmi.biHeight = -self.h          # negative = top-down rows
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0           # BI_RGB
        bits = ctypes.c_void_p()
        self.hbmp = _GDI32.CreateDIBSection(
            screen_dc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
        _USER32.ReleaseDC(None, screen_dc)
        if not self.hbmp or not bits.value:
            raise ctypes.WinError()
        _GDI32.SelectObject(self.mem_dc, self.hbmp)
        raw = (ctypes.c_uint8 * (self.w * self.h * 4)).from_address(bits.value)
        self.buf = np.frombuffer(raw, dtype=np.uint8).reshape(self.h, self.w, 4)

        _USER32.PeekMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND,
                                         wt.UINT, wt.UINT, wt.UINT]
        _USER32.PeekMessageW.restype = wt.BOOL
        _USER32.UpdateLayeredWindow.argtypes = [
            wt.HWND, wt.HDC, ctypes.POINTER(_POINT), ctypes.POINTER(_SIZE),
            wt.HDC, ctypes.POINTER(_POINT), wt.DWORD,
            ctypes.POINTER(_BLENDFUNCTION), wt.DWORD]
        _USER32.UpdateLayeredWindow.restype = wt.BOOL
        self._pump()

    # -- drawing helpers (BGRA, alpha 255 = solid) -----------------------------
    def _rect(self, x1: int, y1: int, x2: int, y2: int, color, t: int = 2) -> None:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.w - 1, x2), min(self.h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return
        b = self.buf
        b[y1:y1 + t, x1:x2 + 1] = color
        b[y2 - t + 1:y2 + 1, x1:x2 + 1] = color
        b[y1:y2 + 1, x1:x1 + t] = color
        b[y1:y2 + 1, x2 - t + 1:x2 + 1] = color

    def _line(self, x0: int, y0: int, x1: int, y1: int, color, t: int = 2) -> None:
        dx, dy = x1 - x0, y1 - y0
        steps = max(abs(dx), abs(dy), 1)
        for i in range(steps + 1):
            x = x0 + dx * i // steps
            y = y0 + dy * i // steps
            if 0 <= x < self.w and 0 <= y < self.h:
                self.buf[max(0, y - t + 1):y + 1,
                         max(0, x - t + 1):x + 1] = color

    def _circle(self, cx: int, cy: int, r: int, color) -> None:
        if r < 2:
            return
        steps = max(48, min(160, r))
        pts = [(int(cx + r * math.cos(2 * math.pi * i / steps)),
                int(cy + r * math.sin(2 * math.pi * i / steps)))
               for i in range(steps + 1)]
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            self._line(ax, ay, bx, by, color, 1)

    def _cross(self, cx: int, cy: int, color, size: int = 9) -> None:
        self._line(cx - size, cy, cx + size, cy, color, 2)
        self._line(cx, cy - size, cx, cy + size, color, 2)

    def _pump(self) -> None:
        msg = wt.MSG()
        while _USER32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
            _USER32.TranslateMessage(ctypes.byref(msg))
            _USER32.DispatchMessageW(ctypes.byref(msg))

    def update(self, center, boxes, target, fov: float) -> None:
        self.buf[:] = 0                                     # clear to transparent
        for x1, y1, x2, y2 in boxes.astype(int):
            self._rect(x1, y1, x2, y2, (0, 128, 255, 255))
        self._circle(int(center[0]), int(center[1]), int(fov), (0, 200, 0, 255))
        if target:
            self._cross(int(target[0]), int(target[1]), (0, 0, 255, 255))
        pt_src, pt_dst = _POINT(0, 0), _POINT(self.x, self.y)
        size = _SIZE(self.w, self.h)
        blend = _BLENDFUNCTION(0, 0, 255, 1)   # AC_SRC_OVER, AC_SRC_ALPHA
        screen_dc = _USER32.GetDC(None)
        _USER32.UpdateLayeredWindow(
            self.hwnd, screen_dc, ctypes.byref(pt_dst), ctypes.byref(size),
            self.mem_dc, ctypes.byref(pt_src), 0, ctypes.byref(blend), 2)
        _USER32.ReleaseDC(None, screen_dc)
        self._pump()

    def close(self) -> None:
        try:
            _USER32.DestroyWindow(self.hwnd)
            _GDI32.DeleteObject(self.hbmp)
            _GDI32.DeleteDC(self.mem_dc)
        except Exception:
            pass



def run_live(cfg: configparser.ConfigParser, scan: bool = False) -> None:
    aim = cfg["aim"]
    aim_key = parse_key(aim["key"])
    toggle_key = parse_key(aim["toggle_key"])
    conf_thr = aim.getfloat("conf")
    fov = aim.getfloat("fov")
    if fov <= 0:
        fov = 400.0            # never lock every target out with a zero radius
    smooth = aim.getfloat("smooth")
    gain = aim.getfloat("gain")
    head_frac = aim.getfloat("head_frac")
    lead_frames = aim.getint("lead")
    allowed = {int(c.strip()) for c in aim["classes"].split(",") if c.strip()}
    debug = cfg["debug"].getboolean("overlay")
    mode = cfg["detect"].get("mode", "color").strip().lower()
    low, high = hsv_bounds(cfg["detect"]["low"], cfg["detect"]["high"])
    min_area = cfg["detect"].getint("min_area")
    max_area = cfg["detect"].getint("max_area")
    max_fill = cfg["detect"].getfloat("max_fill")
    region = None
    raw_region = cfg["capture"].get("region", "").strip()
    if raw_region:
        try:
            l, t, r, b = (int(v) for v in raw_region.split(","))
            region = (l, t, r - l, b - t)
        except ValueError:
            print(f"bad capture.region {raw_region!r}, using full screen")

    session = input_name = None
    if mode == "yolo":
        session, input_name = open_session(app_dir() / cfg["model"]["path"])
    else:
        mode = "color"
    camera = create_camera(cfg)
    if session:
        print(f"providers={session.get_providers()}")
    print(f"live | detect={mode} toggle={key_name(toggle_key)} "
          f"aim={key_name(aim_key)} fov={fov} gain={gain} END=quit")

    overlay = None
    overlay_off = False
    armed, prev_toggle, prev_target = True, False, None
    fps_t, fps_n = time.perf_counter(), 0
    tune_t = time.perf_counter()

    while True:
        if key_down(VK_END) or STOP_EVENT.is_set():
            STOP_EVENT.clear()
            camera.stop()
            if overlay:
                overlay.close()
            print("\nstopped")
            return

        toggled = key_down(toggle_key)
        if toggled and not prev_toggle:
            armed = not armed
            prev_target = None
            print("\nARMED" if armed else "\nSAFE")
        prev_toggle = toggled

        if not armed:
            time.sleep(0.004)
            continue

        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.002)
            continue

        h, w = frame.shape[:2]
        center = (w * 0.5, h * 0.5)

        # panel sliders write straight into cfg: pick the new values up here
        # so aim/detect tuning applies without restarting the loop
        if time.perf_counter() - tune_t > 0.25:
            tune_t = time.perf_counter()
            try:
                smooth = cfg["aim"].getfloat("smooth")
                gain = cfg["aim"].getfloat("gain")
                fov = cfg["aim"].getfloat("fov")
                if fov <= 0:
                    fov = 400.0
                conf_thr = cfg["aim"].getfloat("conf")
                head_frac = cfg["aim"].getfloat("head_frac")
                aim_key = parse_key(cfg["aim"]["key"])
                toggle_key = parse_key(cfg["aim"]["toggle_key"])
                if mode == "color":
                    low, high = hsv_bounds(cfg["detect"]["low"],
                                           cfg["detect"]["high"])
                    min_area = cfg["detect"].getint("min_area")
                    max_area = cfg["detect"].getint("max_area")
                    max_fill = cfg["detect"].getfloat("max_fill")
            except (ValueError, KeyError):
                pass   # half-typed field: keep the previous working values

        if (scan or debug) and overlay is None and not overlay_off:
            try:
                ox, oy = (region[0], region[1]) if region else (0, 0)
                overlay = Overlay(ox, oy, w, h)
                print(f"overlay {w}x{h} at {ox},{oy} (transparent, click-through, topmost)")
            except Exception as exc:
                overlay_off = True
                print(f"overlay failed: {exc}")

        if not scan and not key_down(aim_key):
            prev_target = None
            time.sleep(0.004)
            continue

        if mode == "color":
            rows = color_rows(frame, low, high, min_area, max_area, max_fill)
        else:
            canvas, r, dx, dy = letterbox(frame)
            blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            raw = session.run(None, {input_name: blob})[0]
            rows = decode(raw)
            if rows.size:
                rows[:, [0, 2]] = (rows[:, [0, 2]] - dx) / r
                rows[:, [1, 3]] = (rows[:, [1, 3]] - dy) / r

        target, boxes, prev_state = select_target(rows, center, fov, allowed, conf_thr, head_frac, prev_target)
        prev_target = prev_state

        if scan:
            # no mouse input, just paint boxes on the game overlay
            if overlay:
                overlay.update(center, boxes, target, fov)
            fps_n += 1
            now = time.perf_counter()
            if now - fps_t >= 1.0:
                fps = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0
                print(f"\rfps={fps:.0f} boxes={len(boxes)}   ", flush=True)
            time.sleep(0.002)
            continue

        if target is None:
            prev_target = None
        else:
            # Exponential smoothing to filter out contour noise/jitter
            if prev_target is not None:
                smooth_t = (prev_target[0] * 0.35 + target[0] * 0.65,
                            prev_target[1] * 0.35 + target[1] * 0.65)
            else:
                smooth_t = target

            aim_at = lead_pos(smooth_t, prev_target, lead_frames) if lead_frames > 0 else smooth_t
            raw_dx = aim_at[0] - center[0]
            raw_dy = aim_at[1] - center[1]
            dist_center = math.hypot(raw_dx, raw_dy)

            # Deadzone: if crosshair is within 3 pixels of target, lock still (no jitter!)
            if dist_center <= 3.0:
                mdx, mdy = 0.0, 0.0
            else:
                mdx = raw_dx * (1.0 - smooth)
                mdy = raw_dy * (1.0 - smooth)

            counts_x = max(-w // 3, min(w // 3, round(mdx * gain)))
            counts_y = max(-h // 3, min(h // 3, round(mdy * gain)))
            if counts_x or counts_y:
                move_mouse(counts_x, counts_y)
            prev_target = smooth_t

        if debug and overlay:
            overlay.update(center, boxes, target, fov)

        fps_n += 1
        now = time.perf_counter()
        if now - fps_t >= 1.0:
            fps = fps_n / (now - fps_t)
            fps_t, fps_n = now, 0
            print(f"\rfps={fps:.0f} boxes={len(boxes)}   ", end="", flush=True)


def edit_settings(cfg: configparser.ConfigParser) -> None:
    while True:
        print("\n-- settings (b=back, saves on change) --")
        for i, (sec, key, _, desc) in enumerate(TUNE_FIELDS, 1):
            print(f" {i:2}) {sec}.{key} = {cfg[sec][key]}   ({desc})")
        pick = input("field #: ").strip().lower()
        if pick == "b":
            return
        if not pick.isdigit() or not 1 <= int(pick) <= len(TUNE_FIELDS):
            continue
        sec, key, cast, _ = TUNE_FIELDS[int(pick) - 1]
        new = input(f"{sec}.{key} [{cfg[sec][key]}]: ").strip()
        if not new:
            continue
        try:
            value = str(cast(new))
        except ValueError:
            print("bad value")
            continue
        cfg.set(sec, key, value)
        save_config(cfg)
        print(f"saved {sec}.{key}={value}")


def flush_stdin() -> None:
    """Drop keystrokes typed while the app was still loading.

    One-file builds show a blank console for tens of seconds; anything typed
    meanwhile waits in the input buffer and the first menu prompt would eat it,
    launching straight into live mode before the user even saw the menu.
    """
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getwch()
    except Exception:
        pass


def menu(cfg: configparser.ConfigParser) -> None:
    flush_stdin()
    while True:
        print(f"""
== SkayNn's FramebyFrame ==
 model : {cfg['model']['path']}
 keys  : aim {key_show(cfg['aim']['key'])} hold | toggle {key_show(cfg['aim']['toggle_key'])} | END quits live
 1) start live
 2) selftest
 3) settings
 4) open aimbot.ini
 5) exit""")
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice == "1":
            run_live(cfg)
        elif choice == "2":
            selftest(cfg)
        elif choice == "3":
            edit_settings(cfg)
        elif choice == "4":
            if ini_path().exists():
                os.startfile(ini_path())   # windows shell: opens default ini editor
        elif choice == "5":
            return


def gui(cfg: configparser.ConfigParser) -> None:
    """Ultra-premium modern gaming panel built with CustomTkinter."""
    import customtkinter as ctk
    import tkinter as tk
    from tkinter import colorchooser, simpledialog
    import colorsys
    import json
    import queue
    import re
    import traceback

    # Initialize CustomTkinter styling
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("SkayNn's FramebyFrame")
    root.geometry("980x750")
    root.minsize(920, 700)
    root.configure(fg_color="#0a0c10")

    # Raise to front initially
    root.attributes("-topmost", True)
    root.after(1200, lambda: root.attributes("-topmost", False))
    root.lift()

    # Log queue & stdout redirection
    log_q: "queue.Queue[str]" = queue.Queue()

    class _Sink:
        def write(self, text: str) -> None:
            t = text.strip() if text else ""
            if t:
                log_q.put(t)

        def flush(self) -> None:
            pass

    sys.stdout = _Sink()
    sys.stderr = _Sink()

    # Hide console window if launched directly
    try:
        ctypes.windll.user32.ShowWindow(
            ctypes.windll.kernel32.GetConsoleWindow(), 0)
    except Exception:
        pass

    state = {"worker": None}
    status_text = tk.StringVar(value="IDLE")

    def presets_path() -> Path:
        local = app_dir() / "colors.json"
        if local.exists():
            return local
        bundled = bundle_dir() / "colors.json"
        if bundled.exists():
            try:
                local.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
                return local
            except Exception:
                return bundled
        return local

    def load_presets() -> dict:
        try:
            data = json.loads(presets_path().read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def store_presets(data: dict) -> None:
        try:
            presets_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"presets error: {exc}")

    def log_msg(msg: str):
        log_q.put(msg)

    def launch(fn, label: str) -> None:
        worker = state["worker"]
        if worker and worker.is_alive():
            log_msg("Already running! Press STOP first.")
            return
        STOP_EVENT.clear()
        status_text.set(f"RUNNING: {label.upper()}")
        if label == "live":
            status_pill.configure(text="● RUNNING LIVE", fg_color="#064e3b", text_color="#34d399")
        elif label == "scan":
            status_pill.configure(text="● SCAN OVERLAY", fg_color="#1e3a5f", text_color="#38bdf8")
        else:
            status_pill.configure(text=f"● {label.upper()}", fg_color="#1e2433", text_color="#cbd5e1")

        def run() -> None:
            try:
                fn()
            except SystemExit as exc:
                log_msg(f"Process stopped: {exc}")
            except Exception:
                log_msg(traceback.format_exc())
            finally:
                root.after(0, lambda: (
                    status_text.set("IDLE"),
                    status_pill.configure(text="● IDLE", fg_color="#1e2433", text_color="#94a3b8")
                ))

        thread = threading.Thread(target=run, daemon=True)
        state["worker"] = thread
        thread.start()

    def stop_now() -> None:
        STOP_EVENT.set()
        status_text.set("STOPPING...")
        status_pill.configure(text="● STOPPING", fg_color="#451a1a", text_color="#f87171")
        log_msg("Stopping background worker...")

    def sample_now() -> None:
        log_msg("Sampling outline color from screen (~3s)...")
        status_pill.configure(text="● SAMPLING", fg_color="#3b1d54", text_color="#d8b4fe")

        def work():
            ok = sample_color(cfg)
            if ok:
                update_color_preview()
                log_msg("Color sampled successfully from screen!")
            else:
                log_msg("Could not sample color. Point crosshair at enemy outline and retry.")
            root.after(0, lambda: status_pill.configure(text="● IDLE", fg_color="#1e2433", text_color="#94a3b8"))

        threading.Thread(target=work, daemon=True).start()

    # ---- 1. HEADER (BRANDING & STATUS) -----------------------------------
    head = ctk.CTkFrame(root, fg_color="#111420", corner_radius=12, height=65)
    head.pack(fill="x", padx=16, pady=(12, 6))

    brand_frame = ctk.CTkFrame(head, fg_color="transparent")
    brand_frame.pack(side="left", padx=18, pady=10)
    ctk.CTkLabel(brand_frame, text="SkayNn's FramebyFrame", font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
                 text_color="#00f0ff").pack(side="left")
    ctk.CTkLabel(brand_frame, text=" v2.5 PRO", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
                 text_color="#64748b").pack(side="left", padx=(8, 0), pady=(4, 0))

    status_pill = ctk.CTkLabel(head, text="● IDLE", fg_color="#1e2433", corner_radius=12,
                               font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                               text_color="#94a3b8", padx=16, pady=6)
    status_pill.pack(side="right", padx=18, pady=12)

    # ---- 2. TOP ACTION TOOLBAR -------------------------------------------
    bar = ctk.CTkFrame(root, fg_color="#111420", corner_radius=12, height=54)
    bar.pack(fill="x", padx=16, pady=4)

    ctk.CTkButton(bar, text="▶ START LIVE", fg_color="#00f0ff", hover_color="#38bdf8",
                  text_color="#05070a", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                  height=38, corner_radius=8, width=130,
                  command=lambda: launch(lambda: run_live(cfg), "live")).pack(side="left", padx=(12, 6), pady=8)

    ctk.CTkButton(bar, text="👁 SCAN OVERLAY", fg_color="#1a2030", hover_color="#283247",
                  text_color="#e2e8f0", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                  height=38, corner_radius=8, width=130,
                  command=lambda: launch(lambda: run_live(cfg, scan=True), "scan")).pack(side="left", padx=6, pady=8)

    ctk.CTkButton(bar, text="🎨 SAMPLE COLOR", fg_color="#1a2030", hover_color="#283247",
                  text_color="#e2e8f0", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                  height=38, corner_radius=8, width=130,
                  command=sample_now).pack(side="left", padx=6, pady=8)

    ctk.CTkButton(bar, text="🧪 SELFTEST", fg_color="#1a2030", hover_color="#283247",
                  text_color="#e2e8f0", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                  height=38, corner_radius=8, width=110,
                  command=lambda: launch(lambda: selftest(cfg), "selftest")).pack(side="left", padx=6, pady=8)

    ctk.CTkButton(bar, text="⏹ STOP", fg_color="#ef4444", hover_color="#dc2626",
                  text_color="#ffffff", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
                  height=38, corner_radius=8, width=90,
                  command=stop_now).pack(side="left", padx=6, pady=8)

    # ---- 3. MODERN TABVIEW -----------------------------------------------
    tabview = ctk.CTkTabview(root, fg_color="#111420", segmented_button_fg_color="#07090e",
                             segmented_button_selected_color="#00f0ff", segmented_button_selected_hover_color="#38bdf8",
                             text_color="#ffffff", corner_radius=12)
    tabview.pack(fill="both", expand=True, padx=16, pady=(6, 4))

    tab_aim = tabview.add("  🎯 AIMBOT ENGINE  ")
    tab_vis = tabview.add("  🎨 VISION & COLOR  ")
    tab_set = tabview.add("  ⚙️ SYSTEM & PRESETS  ")

    # ================= TAB 1: AIMBOT ENGINE ===============================
    aim_grid = ctk.CTkFrame(tab_aim, fg_color="transparent")
    aim_grid.pack(fill="both", expand=True, padx=10, pady=8)
    aim_grid.columnconfigure(0, weight=1)
    aim_grid.columnconfigure(1, weight=1)

    # Left Card: Sliders
    c_left = ctk.CTkFrame(aim_grid, fg_color="#171b2b", corner_radius=10, border_width=1, border_color="#242b40")
    c_left.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)

    ctk.CTkLabel(c_left, text="TRACKING SENSITIVITY & FOV", font=ctk.CTkFont(size=13, weight="bold"),
                 text_color="#00f0ff").pack(anchor="w", padx=16, pady=(12, 6))

    slider_badges = {}

    def make_slider_row(parent, title, key, lo, hi, fmt):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=4)
        top = ctk.CTkFrame(row, fg_color="transparent")
        top.pack(fill="x")
        ctk.CTkLabel(top, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="#cbd5e1").pack(side="left")
        val_lbl = ctk.CTkLabel(top, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00f0ff")
        val_lbl.pack(side="right")
        slider_badges[key] = (val_lbl, fmt)

        def on_change(v):
            val_text = fmt.format(v)
            val_lbl.configure(text=val_text)
            cfg.set("aim", key, f"{v:.2f}" if "." in fmt else str(int(v)))

        cur = float(cfg["aim"].get(key, "0"))
        slider = ctk.CTkSlider(row, from_=lo, to=hi, command=on_change,
                               button_color="#00f0ff", button_hover_color="#38bdf8",
                               progress_color="#00f0ff", fg_color="#242b40", height=18)
        slider.set(cur)
        val_lbl.configure(text=fmt.format(cur))
        slider.bind("<ButtonRelease-1>", lambda e: save_config(cfg))
        slider.pack(fill="x", pady=(2, 6))

    make_slider_row(c_left, "FOV Radius (Engagement Zone)", "fov", 50, 800, "{:.0f} px")
    make_slider_row(c_left, "Smooth (Aim Gliding)", "smooth", 0.0, 0.95, "{:.2f}")
    make_slider_row(c_left, "Gain (Mouse Sensitivity Multiplier)", "gain", 0.1, 2.5, "{:.2f}x")
    make_slider_row(c_left, "Head Height Offset (Lower = Forehead)", "head_frac", 0.05, 0.40, "{:.2f}")

    # Right Card: Keybinds & Status
    c_right = ctk.CTkFrame(aim_grid, fg_color="#171b2b", corner_radius=10, border_width=1, border_color="#242b40")
    c_right.grid(row=0, column=1, sticky="nsew", padx=8, pady=6)

    ctk.CTkLabel(c_right, text="KEYBINDS & CONTROLS", font=ctk.CTkFont(size=13, weight="bold"),
                 text_color="#00f0ff").pack(anchor="w", padx=16, pady=(12, 10))

    capture_state = {"binding": None}

    def start_bind(k, btn):
        capture_state["binding"] = (k, btn)
        btn.configure(text="[ PRESS KEY... ]", fg_color="#38bdf8", text_color="#05070a")
        root.focus_set()

    def make_key_card(parent, title, key_field):
        f = ctk.CTkFrame(parent, fg_color="#10131e", corner_radius=8, height=50)
        f.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(f, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color="#cbd5e1").pack(side="left", padx=12, pady=10)
        cur_k = key_show(cfg["aim"].get(key_field, "e")).upper()
        btn = ctk.CTkButton(f, text=f"[ {cur_k} ]", width=120, height=32, corner_radius=6,
                            fg_color="#1e2433", hover_color="#283247",
                            font=ctk.CTkFont(family="Consolas", size=13, weight="bold"), text_color="#00f0ff")
        btn.configure(command=lambda k=key_field, b=btn: start_bind(k, b))
        btn.pack(side="right", padx=12, pady=8)
        return btn

    aim_btn = make_key_card(c_right, "Aim Lock Key (Hold to Aim)", "key")
    toggle_btn = make_key_card(c_right, "Toggle Arm / Disarm Key", "toggle_key")

    tip_box = ctk.CTkFrame(c_right, fg_color="#10131e", corner_radius=8)
    tip_box.pack(fill="both", expand=True, padx=16, pady=(10, 14))
    ctk.CTkLabel(tip_box, text="⚡ PRO AIM TIPS", font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8").pack(anchor="w", padx=12, pady=(8, 2))
    ctk.CTkLabel(tip_box, text="• Hold E while enemy is inside circle to lock.\n• Anti-jitter & Head Centroid filter are active.\n• Distant targets are tracked automatically.",
                 font=ctk.CTkFont(size=11), text_color="#94a3b8", justify="left").pack(anchor="w", padx=12, pady=(0, 8))

    def on_window_key(evt):
        bind_info = capture_state["binding"]
        if not bind_info:
            return
        k_field, btn = bind_info
        sym = (evt.keysym or "").lower()
        vk = getattr(evt, "keycode", 0) or 0
        if sym in ("shift", "control", "alt", "caps_lock", "escape", "space",
                   "tab", "return", "backspace", "insert", "delete",
                   "home", "end", "prior", "next", "up", "down",
                   "left", "right", "num_lock", "scroll_lock", "pause"):
            name = {"caps_lock": "caps", "escape": "esc", "prior": "pgup",
                    "next": "pgdn", "num_lock": "numlock",
                    "scroll_lock": "scrolllock", "control": "ctrl"}.get(sym, sym)
        elif sym.startswith("f") and sym[1:].isdigit():
            name = sym
        elif len(sym) == 1:
            name = sym
        elif vk:
            name = key_name(vk)
        else:
            return
        try:
            parse_key(name)
        except ValueError:
            return
        cfg.set("aim", k_field, name)
        save_config(cfg)
        capture_state["binding"] = None
        btn.configure(text=f"[ {name.upper()} ]", fg_color="#1e2433", text_color="#00f0ff")
        log_msg(f"Key assigned: {k_field} = {name}")
        return "break"

    root.bind("<KeyPress>", on_window_key)

    # ================= TAB 2: VISION & OUTLINE ============================
    vis_grid = ctk.CTkFrame(tab_vis, fg_color="transparent")
    vis_grid.pack(fill="both", expand=True, padx=10, pady=8)
    vis_grid.columnconfigure(0, weight=1)
    vis_grid.columnconfigure(1, weight=1)

    v_left = ctk.CTkFrame(vis_grid, fg_color="#171b2b", corner_radius=10, border_width=1, border_color="#242b40")
    v_left.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)

    ctk.CTkLabel(v_left, text="DETECTOR & PRESET PALETTE", font=ctk.CTkFont(size=13, weight="bold"),
                 text_color="#00f0ff").pack(anchor="w", padx=16, pady=(12, 6))

    # Detector mode toggle
    mode_frame = ctk.CTkFrame(v_left, fg_color="#10131e", corner_radius=8)
    mode_frame.pack(fill="x", padx=16, pady=4)
    ctk.CTkLabel(mode_frame, text="DETECTOR ENGINE:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#cbd5e1").pack(side="left", padx=12, pady=10)

    def on_mode_switch(m):
        mode_val = "color" if "COLOR" in m else "yolo"
        cfg.set("detect", "mode", mode_val)
        save_config(cfg)
        log_msg(f"Detector set to: {mode_val.upper()}")

    cur_mode = cfg["detect"].get("mode", "color").upper()
    mode_seg = ctk.CTkSegmentedButton(mode_frame, values=["COLOR OUTLINE", "YOLOv8 AI"], command=on_mode_switch,
                                      selected_color="#00f0ff", selected_hover_color="#38bdf8",
                                      text_color="#05070a", font=ctk.CTkFont(size=12, weight="bold"))
    mode_seg.set("COLOR OUTLINE" if cur_mode == "COLOR" else "YOLOv8 AI")
    mode_seg.pack(side="right", padx=12, pady=8)

    # Palette 12 presets
    PALETTE = [
        ("Neon Yellow", "#ffe81a", (26, 160, 190), (34, 255, 255)),
        ("Acid Green", "#7cff2a", (35, 150, 180), (75, 255, 255)),
        ("Electric Cyan", "#2af0ff", (85, 150, 180), (100, 255, 255)),
        ("Magenta", "#ff3df0", (140, 140, 180), (160, 255, 255)),
        ("Crimson Red", "#ff2a2a", (0, 160, 180), (10, 255, 255)),
        ("Neon Orange", "#ff8a1a", (11, 160, 190), (24, 255, 255)),
        ("Pure White", "#ffffff", (0, 0, 220), (179, 35, 255)),
        ("Violet", "#9a5cff", (125, 130, 180), (145, 255, 255)),
        ("Bright Lime", "#c6ff00", (32, 170, 190), (45, 255, 255)),
        ("Hot Pink", "#ff6ec7", (150, 130, 180), (170, 255, 255)),
        ("Sky Blue", "#4da6ff", (100, 140, 180), (120, 255, 255)),
        ("Vibrant Gold", "#ffb300", (20, 170, 190), (28, 255, 255)),
    ]

    pal_container = ctk.CTkFrame(v_left, fg_color="#10131e", corner_radius=8)
    pal_container.pack(fill="both", expand=True, padx=16, pady=8)
    ctk.CTkLabel(pal_container, text="QUICK ENEMY OUTLINE PALETTE (1-CLICK FIT):", font=ctk.CTkFont(size=11, weight="bold"),
                 text_color="#94a3b8").pack(anchor="w", padx=10, pady=(8, 4))

    pal_grid = ctk.CTkFrame(pal_container, fg_color="transparent")
    pal_grid.pack(padx=8, pady=(0, 8))

    def apply_palette_color(name, hex_c, lo, hi):
        cfg.set("detect", "low", f"{lo[0]},{lo[1]},{lo[2]}")
        cfg.set("detect", "high", f"{hi[0]},{hi[1]},{hi[2]}")
        save_config(cfg)
        update_color_preview()
        log_msg(f"Outline set to: {name}")

    for idx, (pname, phex, plo, phi) in enumerate(PALETTE):
        r, c = idx // 3, idx % 3
        b = ctk.CTkButton(pal_grid, text=pname, fg_color=phex, hover_color="#ffffff",
                          text_color="#000000" if phex in ("#ffe81a", "#7cff2a", "#2af0ff", "#ffffff", "#c6ff00", "#ffb300") else "#ffffff",
                          font=ctk.CTkFont(size=11, weight="bold"), height=28, width=125, corner_radius=6,
                          command=lambda n=pname, h=phex, l=plo, hi=phi: apply_palette_color(n, h, l, hi))
        b.grid(row=r, column=c, padx=3, pady=3)

    # Right Card: Preview & Distant Detection
    v_right = ctk.CTkFrame(vis_grid, fg_color="#171b2b", corner_radius=10, border_width=1, border_color="#242b40")
    v_right.grid(row=0, column=1, sticky="nsew", padx=8, pady=6)

    ctk.CTkLabel(v_right, text="COLOR SWATCH & DISTANT FILTER", font=ctk.CTkFont(size=13, weight="bold"),
                 text_color="#00f0ff").pack(anchor="w", padx=16, pady=(12, 6))

    swatch_box = ctk.CTkFrame(v_right, fg_color="#10131e", corner_radius=8)
    swatch_box.pack(fill="x", padx=16, pady=4)

    swatch_circle = ctk.CTkLabel(swatch_box, text="", width=40, height=40, corner_radius=20, fg_color="#ffe81a")
    swatch_circle.pack(side="left", padx=12, pady=10)

    swatch_info = ctk.CTkLabel(swatch_box, text="Current Color HSV", font=ctk.CTkFont(size=12, weight="bold"), text_color="#cbd5e1")
    swatch_info.pack(side="left", padx=8)

    def pick_color():
        c = colorchooser.askcolor(title="Choose Enemy Outline Color", parent=root)
        if c and c[0]:
            r, g, b = [int(v) for v in c[0]]
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            H = int(h * 179)
            S = int(s * 255)
            V = int(v * 255)
            lo = (max(0, H - 7), max(80, S - 70), max(100, V - 60))
            hi = (min(179, H + 7), 255, 255)
            cfg.set("detect", "low", f"{lo[0]},{lo[1]},{lo[2]}")
            cfg.set("detect", "high", f"{hi[0]},{hi[1]},{hi[2]}")
            save_config(cfg)
            update_color_preview()
            log_msg(f"Custom color chosen: H={H} S={S} V={V}")

    ctk.CTkButton(swatch_box, text="🎨 PICK WHEEL", width=100, height=32, corner_radius=6,
                  fg_color="#1e2433", hover_color="#283247", font=ctk.CTkFont(size=11, weight="bold"),
                  command=pick_color).pack(side="right", padx=12)

    def update_color_preview():
        try:
            lo = parse_hsv(cfg["detect"]["low"])
            hi = parse_hsv(cfg["detect"]["high"])
            avg_h = ((lo[0] + hi[0]) / 2) / 179.0
            avg_s = min(1.0, ((lo[1] + hi[1]) / 2) / 255.0 + 0.2)
            avg_v = min(1.0, ((lo[2] + hi[2]) / 2) / 255.0 + 0.1)
            pr, pg, pb = colorsys.hsv_to_rgb(avg_h, avg_s, avg_v)
            hex_c = f"#{int(pr*255):02x}{int(pg*255):02x}{int(pb*255):02x}"
            swatch_circle.configure(fg_color=hex_c)
            swatch_info.configure(text=f"Outline HSV: {lo[0]}..{hi[0]} (Hex: {hex_c.upper()})")
        except Exception:
            pass

    update_color_preview()

    # Distant enemy & blob tuning
    dist_frame = ctk.CTkFrame(v_right, fg_color="#10131e", corner_radius=8)
    dist_frame.pack(fill="both", expand=True, padx=16, pady=(8, 12))

    ctk.CTkLabel(dist_frame, text="DISTANCE RANGE & BLOB SENSITIVITY:", font=ctk.CTkFont(size=11, weight="bold"),
                 text_color="#38bdf8").pack(anchor="w", padx=12, pady=(10, 2))

    # Min Area slider (lower = extreme distance)
    area_row = ctk.CTkFrame(dist_frame, fg_color="transparent")
    area_row.pack(fill="x", padx=12, pady=4)
    area_top = ctk.CTkFrame(area_row, fg_color="transparent")
    area_top.pack(fill="x")
    ctk.CTkLabel(area_top, text="Min Target Area (Lower = Farther Distance)", font=ctk.CTkFont(size=12), text_color="#cbd5e1").pack(side="left")
    area_lbl = ctk.CTkLabel(area_top, text="", font=ctk.CTkFont(size=12, weight="bold"), text_color="#00f0ff")
    area_lbl.pack(side="right")

    def on_area_change(v):
        val = int(v)
        area_lbl.configure(text=f"{val} px²")
        cfg.set("detect", "min_area", str(val))

    cur_area = int(cfg["detect"].get("min_area", "35"))
    area_slider = ctk.CTkSlider(dist_frame, from_=20, to=400, command=on_area_change,
                                button_color="#00f0ff", button_hover_color="#38bdf8",
                                progress_color="#00f0ff", fg_color="#242b40", height=18)
    area_slider.set(cur_area)
    area_lbl.configure(text=f"{cur_area} px²")
    area_slider.bind("<ButtonRelease-1>", lambda e: save_config(cfg))
    area_slider.pack(fill="x", padx=12, pady=(2, 6))

    ctk.CTkLabel(dist_frame, text="✓ Set to 30-40 px² to detect enemies across the entire map.",
                 font=ctk.CTkFont(size=11), text_color="#94a3b8").pack(anchor="w", padx=12, pady=(0, 8))

    # ================= TAB 3: SYSTEM & PRESETS ============================
    set_card = ctk.CTkFrame(tab_set, fg_color="#171b2b", corner_radius=10, border_width=1, border_color="#242b40")
    set_card.pack(fill="both", expand=True, padx=12, pady=10)

    ctk.CTkLabel(set_card, text="SYSTEM & ENGINE PRESETS", font=ctk.CTkFont(size=14, weight="bold"),
                 text_color="#00f0ff").pack(anchor="w", padx=18, pady=(14, 10))

    # Overlay Switch
    ov_frame = ctk.CTkFrame(set_card, fg_color="#10131e", corner_radius=8)
    ov_frame.pack(fill="x", padx=18, pady=6)
    ctk.CTkLabel(ov_frame, text="Target Overlay / ESP Boxes:", font=ctk.CTkFont(size=12, weight="bold"), text_color="#cbd5e1").pack(side="left", padx=12, pady=10)

    def on_toggle_overlay():
        new_val = str(ov_switch.get())
        cfg.set("debug", "overlay", new_val)
        save_config(cfg)
        log_msg(f"Overlay set to: {new_val}")

    ov_switch = ctk.CTkSwitch(ov_frame, text="Active", command=on_toggle_overlay,
                              progress_color="#00f0ff", button_color="#ffffff",
                              font=ctk.CTkFont(size=12, weight="bold"))
    if cfg["debug"].getboolean("overlay", True):
        ov_switch.select()
    else:
        ov_switch.deselect()
    ov_switch.pack(side="right", padx=12, pady=10)

    # Actions: Open ini / Reload
    act_frame = ctk.CTkFrame(set_card, fg_color="#10131e", corner_radius=8)
    act_frame.pack(fill="x", padx=18, pady=10)
    ctk.CTkButton(act_frame, text="📄 OPEN AIMBOT.INI", height=36, corner_radius=6,
                  fg_color="#1e2433", hover_color="#283247", font=ctk.CTkFont(size=12, weight="bold"),
                  command=lambda: ini_path().exists() and os.startfile(ini_path())).pack(side="left", padx=12, pady=10)

    ctk.CTkButton(act_frame, text="🔄 RELOAD CONFIG", height=36, corner_radius=6,
                  fg_color="#1e2433", hover_color="#283247", font=ctk.CTkFont(size=12, weight="bold"),
                  command=lambda: (cfg.read(ini_path(), encoding="utf-8"), log_msg("Config reloaded!"))).pack(side="left", padx=6, pady=10)

    # ---- 4. BOTTOM LOG TERMINAL ------------------------------------------
    term_frame = ctk.CTkFrame(root, fg_color="#111420", corner_radius=12, height=140)
    term_frame.pack(fill="x", padx=16, pady=(4, 12))

    term_head = ctk.CTkFrame(term_frame, fg_color="transparent")
    term_head.pack(fill="x", padx=12, pady=(6, 2))
    ctk.CTkLabel(term_head, text="LIVE TERMINAL OUTPUT", font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), text_color="#64748b").pack(side="left")

    log_box = ctk.CTkTextbox(term_frame, fg_color="#07090e", text_color="#38bdf8",
                             font=ctk.CTkFont(family="Consolas", size=11), corner_radius=8, height=95)
    log_box.pack(fill="both", expand=True, padx=12, pady=(0, 8))
    log_box.insert("end", f"[{time.strftime('%H:%M:%S')}] SkayNn's FramebyFrame Initialized.\n")
    log_box.configure(state="disabled")

    def drain():
        try:
            while not log_q.empty():
                msg = log_q.get_nowait()
                log_box.configure(state="normal")
                log_box.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
                log_box.see("end")
                log_box.configure(state="disabled")
        except Exception:
            pass
        root.after(100, drain)

    root.after(100, drain)
    root.protocol("WM_DELETE_WINDOW", lambda: (STOP_EVENT.set(), root.destroy()))

    try:
        import pyi_splash
        pyi_splash.close()
    except Exception:
        pass

    root.mainloop()



def _set_dpi_aware() -> None:
    """Line overlay pixels up with screen pixels.

    On a scaled display (125/150%) an unaware process sees logical coordinates
    while dxcam captures physical ones, so boxes drift away from the target.
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor v2
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _close_splash() -> None:
    """Drop the PyInstaller splash once python is up (all modes, not just GUI)."""
    try:
        import pyi_splash
        pyi_splash.close()
    except Exception:
        pass


def main() -> None:
    _set_dpi_aware()
    _close_splash()
    cfg = load_config()
    if "--selftest" in sys.argv:
        selftest(cfg)
    elif "--live" in sys.argv:
        run_live(cfg)
    elif "--scan" in sys.argv:
        run_live(cfg, scan=True)
    elif "--menu" in sys.argv:
        menu(cfg)
    else:
        # bare launch = GUI panel; console menu stays as a fallback
        try:
            gui(cfg)
        except Exception:
            traceback.print_exc()
            menu(cfg)


if __name__ == "__main__":
    main()
