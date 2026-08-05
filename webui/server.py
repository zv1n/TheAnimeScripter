r"""
TAS Web UI — tiny local control panel for preparing raw MKVs (ffmpeg) and
running TheAnimeScripter upscales, with live progress for both.

Runs on the bundled Python 3.14 in this folder. No external web framework:
stdlib http.server + Server-Sent Events for progress. TAS progress is bridged
from TAS's own `--ae` Socket.IO feed via a socketio client.

Launch with TAS-UI.bat (repo root) or:  .\python.exe webui\server.py
"""

import json
import os
import re
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PYTHON = os.path.join(REPO, "python.exe")
MAIN = os.path.join(REPO, "main.py")
FFMPEG = os.path.join(REPO, "ffmpeg_shared", "ffmpeg.exe")
FFPROBE = os.path.join(REPO, "ffmpeg_shared", "ffprobe.exe")
INDEX = os.path.join(HERE, "index.html")

PORT = int(os.environ.get("TAS_UI_PORT", "5001"))
VIDEO_EXTS = (".mkv", ".mp4", ".mov", ".avi", ".m2ts", ".ts", ".webm")
MODEL_EXTS = (".onnx", ".pth", ".pt", ".ckpt", ".safetensors")


def _detect_root(candidates):
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


VIDEO_ROOT = os.environ.get("TAS_VIDEO_ROOT") or _detect_root(
    ["E:/Video", "E:/videos", "E:/Videos"]
)
MODEL_ROOT = os.environ.get("TAS_MODEL_ROOT") or _detect_root(["E:/models", "E:/Models"])

# span-directml first = default. TensorRT is listed but note: it fails to build
# an engine for the FP32 SPAN model on this cu13/TRT11 stack (Conv/Clip node has
# no valid tactic). DirectML runs the .onnx on the NVIDIA GPU via DX12 and works.
UPSCALE_METHODS = [
    "span-directml", "span-openvino", "span-tensorrt", "span-ncnn",
    "shufflecugan-directml", "shufflecugan-tensorrt",
    "open-proteus-tensorrt", "aniscale2-tensorrt", "rtmosr-tensorrt",
    "adore-tensorrt", "shufflespan-tensorrt",
]
# Software encoders first = default. nvenc_* need NVIDIA driver 610+ (this box is
# on 595.97 → nvenc fails to open and TAS pipes into a dead encoder for the whole
# run before reporting "0 bytes"). x265_10bit is software and always works.
ENCODE_METHODS = [
    "x265_10bit", "x265", "x264", "prores",
    "nvenc_h265_10bit", "nvenc_h265", "nvenc_h264", "nvenc_av1",
]

# ---------------------------------------------------------------- job registry

jobs = {}
jobs_lock = threading.Lock()
# The single canonical job per kind the UI binds to. Survives page refreshes:
# the browser calls /api/state on load and reconnects to these.
current = {"prep": None, "tas": None}
_port_counter = [8090]
_port_lock = threading.Lock()


def _next_port():
    with _port_lock:
        p = _port_counter[0]
        _port_counter[0] += 1
        return p


class Job:
    def __init__(self, jid, kind, output):
        self.id = jid
        self.kind = kind
        self.output = output
        self.proc = None
        self.log = []
        self.done = False
        self._lock = threading.Lock()
        self.version = 0
        self.last = {"status": "starting", "progress": 0.0, "output": output}

    def update(self, data):
        with self._lock:
            self.last = {**self.last, **data}
            self.version += 1

    def snapshot(self):
        with self._lock:
            return dict(self.last), self.version


def _new_job(kind, output):
    jid = f"{kind}-{int(time.time() * 1000)}"
    job = Job(jid, kind, output)
    with jobs_lock:
        jobs[jid] = job
        current[kind] = jid
    return job


# ---------------------------------------------------------------- ffmpeg probe


def _probe_duration(path):
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", path],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def _parse_out_time(cur):
    us = cur.get("out_time_us") or cur.get("out_time_ms")
    if us and us.lstrip("-").isdigit():
        return int(us) / 1_000_000
    t = cur.get("out_time")
    if t and ":" in t:
        try:
            h, m, s = t.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------- prepare (ffmpeg)


# DAR-correct, square-pixel scale shared by every cadence path.
_SCALE = "scale='trunc(ih*dar/2)*2:ih':flags=lanczos,setsar=1"
CADENCES = ("auto", "film24", "telecine", "progressive", "interlaced")


def _prep_plan(cadence):
    """Return (video_filter, [extra output args]) for a cadence.

    Every path pins CFR so the source's pulldown/VFR timing can't leak through
    and cause judder downstream — the failure mode where 24p frames carried at
    29.97 pulldown timing get spaced 33/50/33ms instead of a clean 41.7ms.
    """
    if cadence == "telecine":    # hard 3:2 pulldown baked into 29.97 frames
        return (f"fieldmatch,yadif=deint=interlaced,decimate,{_SCALE}", [])
    if cadence == "film24":      # 24p film / soft telecine → clean CFR 23.976
        return (_SCALE, ["-fps_mode", "cfr", "-r", "24000/1001"])
    if cadence == "interlaced":  # true interlaced video → deinterlace, keep rate
        return (f"yadif,{_SCALE}", ["-fps_mode", "cfr"])
    return (_SCALE, ["-fps_mode", "cfr"])   # progressive → native rate, CFR


def _detect_cadence(path):
    """Classify cadence from a mid-file sample.

    Key signal: the effective *decoded* frame rate. 24p film (native or soft
    telecine) decodes at ~24fps even when the container is labelled 29.97, so
    it must be pinned to CFR 23.976 rather than decimated. Hard 3:2 telecine
    decodes at ~30fps with repeated fields; true interlaced shows TFF/BFF.
    """
    dur = _probe_duration(path) or 0
    ss = max(0, dur * 0.4)

    eff = None
    try:
        err = subprocess.run(
            [FFMPEG, "-hide_banner", "-ss", str(ss), "-i", path, "-t", "8",
             "-an", "-map", "0:v:0", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120).stderr
        m = re.findall(r"frame=\s*(\d+)", err)
        if m:
            eff = int(m[-1]) / 8.0
    except Exception:
        pass

    tff = bff = prog = repeated = 0
    try:
        out = subprocess.run(
            [FFMPEG, "-hide_banner", "-ss", str(ss), "-i", path, "-vf", "idet",
             "-frames:v", "400", "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=180).stderr

        def line(label):
            for ln in out.splitlines():
                if label in ln:
                    return ln
            return ""

        def num(s, key):
            m = re.search(key + r":\s*(\d+)", s)
            return int(m.group(1)) if m else 0

        ml = line("Multi frame detection")
        rf = line("Repeated Fields")
        tff, bff, prog = num(ml, "TFF"), num(ml, "BFF"), num(ml, "Progressive")
        repeated = num(rf, "Top") + num(rf, "Bottom")
    except Exception:
        pass

    total = max(1, tff + bff + prog)
    if eff is not None and eff < 26:     # ~24fps decode ⇒ 24p film / soft telecine
        cadence, why = "film24", f"decoded {eff:.2f} fps ⇒ 24p film / soft telecine"
    elif repeated / total > 0.05:        # 3:2 pulldown leaves repeated fields
        cadence, why = "telecine", "repeated fields ⇒ hard 3:2 telecine"
    elif (tff + bff) / total > 0.10:
        cadence, why = "interlaced", "TFF/BFF fields ⇒ interlaced video"
    else:
        cadence, why = "progressive", "progressive, ~30 fps"
    return {
        "cadence": cadence, "why": why,
        "decodedFps": round(eff, 3) if eff is not None else None,
        "progressive": prog, "interlaced": tff + bff,
        "repeatedFields": repeated, "sampleFrames": total,
    }


def _measure_timing(path, n=48):
    """Sample frame PTS mid-file and report whether spacing is constant (CFR).
    This is the direct 'is the timing ACTUALLY correct' check — a VFR/pulldown
    leak shows up here as mixed 33/50/66 ms deltas."""
    dur = _probe_duration(path) or 0
    ss = max(0, dur * 0.4)
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-read_intervals", f"{ss}%+#{n}",
             "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=90).stdout
        ts = sorted(float(x) for x in re.findall(r"\d+\.\d+", out))
        deltas = [round(ts[i + 1] - ts[i], 4)
                  for i in range(len(ts) - 1) if ts[i + 1] > ts[i]]
        if len(deltas) < 6:
            return None
        deltas.sort()
        lo, hi, med = deltas[0], deltas[-1], deltas[len(deltas) // 2]
        return {"cfr": (hi - lo) <= 0.004, "minDelta": lo, "maxDelta": hi,
                "fps": round(1.0 / med, 3) if med > 0 else None}
    except Exception:
        return None


def start_prepare(input_path, cadence="auto"):
    base, _ = os.path.splitext(input_path)
    output = base + "-prepped.mkv"
    job = _new_job("prep", output)
    threading.Thread(
        target=_run_prepare, args=(job, input_path, output, cadence), daemon=True
    ).start()
    return job


def _fps_to_float(s):
    try:
        if s and "/" in s:
            a, b = s.split("/")
            return round(int(a) / int(b), 3) if int(b) else None
        return round(float(s), 3) if s else None
    except Exception:
        return None


def _probe_video(path):
    """Duration, fps, resolution, SAR/DAR (fast) + exact frame count (slow:
    counts packets, which reads the whole index)."""
    info = {"duration": None, "fps": None, "width": None, "height": None,
            "sar": None, "dar": None, "frames": None}
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate,avg_frame_rate,width,height,"
             "sample_aspect_ratio,display_aspect_ratio",
             "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=60)
        d = json.loads(r.stdout)
        st = (d.get("streams") or [{}])[0]
        fmt = d.get("format") or {}
        info["width"], info["height"] = st.get("width"), st.get("height")
        info["sar"] = st.get("sample_aspect_ratio")
        info["dar"] = st.get("display_aspect_ratio")
        info["duration"] = float(fmt["duration"]) if fmt.get("duration") else None
        info["fps"] = _fps_to_float(st.get("avg_frame_rate") or st.get("r_frame_rate"))
    except Exception:
        pass
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets",
             "-of", "default=nk=1:nw=1", path],
            capture_output=True, text=True, timeout=1800)
        info["frames"] = int(r.stdout.strip())
    except Exception:
        pass
    return info


def _probe_meta(path):
    """Fast container metadata (no frame counting) for the detect preview."""
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate,width,height,display_aspect_ratio",
             "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=30)
        d = json.loads(r.stdout)
        st = (d.get("streams") or [{}])[0]
        fmt = d.get("format") or {}
        return {"containerFps": _fps_to_float(st.get("r_frame_rate")),
                "width": st.get("width"), "height": st.get("height"),
                "dar": st.get("display_aspect_ratio"),
                "duration": float(fmt["duration"]) if fmt.get("duration") else None}
    except Exception:
        return {}


def _compare_source_output(src, out, cadence="auto"):
    """Row list comparing source vs prepared for the UI. ok=True (match),
    False (mismatch → flag), None (informational, expected to differ).

    Frame rate is compared as the EFFECTIVE rate (frames÷duration), not the
    container label — soft-telecine sources are mislabelled 29.97 while really
    24p, so the label would false-alarm. The output is also checked for CFR
    consistency (label ≈ effective): that's exactly what a VFR/pulldown-timing
    leak fails (frames tagged 29.97 but spaced for 23.976 → judder)."""
    s, o = _probe_video(src), _probe_video(out)

    def effrate(i):
        return (i["frames"] / i["duration"]) \
            if (i["frames"] and i["duration"]) else None

    se, oe = effrate(s), effrate(o)

    def dur(v):
        return f"{v:.2f} s" if isinstance(v, (int, float)) else "—"

    def rate(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    def res(i):
        return f'{i["width"]}x{i["height"]}' if i["width"] else "—"

    dur_ok = bool(s["duration"] and o["duration"]
                  and abs(s["duration"] - o["duration"]) <= 0.5)
    # frame count only expected to match when we're not decimating (telecine).
    # CFR retiming can legitimately shift a couple of frames at the boundaries,
    # so allow a small tolerance (still catches gross loss like decimate's ~20%).
    ftol = max(4, round(0.5 * (oe or 24)))
    frames_match = s["frames"] is not None and o["frames"] is not None \
        and abs(s["frames"] - o["frames"]) <= ftol
    frames_check = None if cadence == "telecine" else frames_match
    rate_match = bool(se and oe and abs(se - oe) / se < 0.02)
    cfr_ok = bool(oe and o["fps"] and abs(oe - o["fps"]) / oe < 0.02)
    rate_ok = rate_match and cfr_ok
    dar_ok = bool(s["dar"] and o["dar"] and s["dar"] == o["dar"])

    # Direct proof the OUTPUT timing is actually constant (not just inferred).
    ts, to = _measure_timing(src), _measure_timing(out)
    timing_ok = bool(to and to["cfr"])

    def timing_disp(t):
        if not t:
            return "—"
        if t["cfr"]:
            return f"constant {t['maxDelta'] * 1000:.0f}ms"
        return f"VFR {t['minDelta'] * 1000:.0f}–{t['maxDelta'] * 1000:.0f}ms"

    out_rate = rate(oe) + ("" if cfr_ok else f" ⚠tagged {rate(o['fps'])}")
    rows = [
        {"label": "Duration", "src": dur(s["duration"]), "out": dur(o["duration"]),
         "ok": dur_ok},
        {"label": "Frames",
         "src": s["frames"] if s["frames"] is not None else "—",
         "out": o["frames"] if o["frames"] is not None else "—",
         "ok": frames_check},
        {"label": "Frame rate", "src": rate(se), "out": out_rate, "ok": rate_ok},
        {"label": "Timing", "src": timing_disp(ts), "out": timing_disp(to),
         "ok": timing_ok},
        {"label": "Aspect (DAR)", "src": s["dar"] or "—", "out": o["dar"] or "—",
         "ok": dar_ok},
        {"label": "Resolution", "src": res(s), "out": res(o), "ok": None},
    ]
    all_ok = dur_ok and rate_ok and timing_ok and dar_ok \
        and frames_check is not False
    return {"rows": rows, "allOk": all_ok}


def _run_prepare(job, input_path, output, cadence):
    duration = _probe_duration(input_path)
    detect = None
    if cadence == "auto":
        job.update({"status": "detecting", "input": input_path, "output": output,
                    "totalDuration": duration, "progress": 0.0,
                    "note": "Detecting cadence…"})
        detect = _detect_cadence(input_path)
        cadence = detect["cadence"]
    job.update({"status": "processing", "input": input_path, "output": output,
                "totalDuration": duration, "progress": 0.0, "cadence": cadence,
                "detect": detect, "note": None})
    vf, rate_args = _prep_plan(cadence)
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-i", input_path,
        "-map", "0:v:0", "-vf", vf, *rate_args,
        "-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1",
        "-progress", "pipe:1", "-nostats", output,
    ]

    def _verify():
        job.update({"status": "comparing", "progress": 100.0,
                    "note": "Verifying output vs source (counting frames)…"})
        return {"compare": _compare_source_output(input_path, output, cadence)}

    _run_ffmpeg_with_progress(job, cmd, output, duration, on_success=_verify)


def start_verify(source, output):
    """Re-run the source-vs-output comparison on an already-prepared file,
    with no re-encode. Reuses the 'prep' slot so the UI renders it identically."""
    job = _new_job("prep", output)

    def run():
        duration = _probe_duration(source)
        job.update({"status": "detecting", "input": source, "output": output,
                    "totalDuration": duration, "progress": 100.0,
                    "note": "Detecting cadence…", "verifyOnly": True})
        cadence = _detect_cadence(source)["cadence"]
        job.update({"status": "comparing", "cadence": cadence, "progress": 100.0,
                    "note": "Verifying output vs source (counting frames)…"})
        try:
            cmp = _compare_source_output(source, output, cadence)
            job.update({"status": "completed", "progress": 100.0, "compare": cmp})
        except Exception as e:
            job.update({"status": "failed", "error": str(e)})
        job.done = True

    threading.Thread(target=run, daemon=True).start()
    return job


def _run_ffmpeg_with_progress(job, cmd, output, duration, on_success=None):
    """Run an ffmpeg command, streaming -progress into the job snapshot.

    on_success: optional callable run after a clean exit; whatever dict it
    returns is merged into the terminal 'completed' snapshot (used to attach a
    source-vs-output comparison). It may emit its own interim status updates.
    """
    try:
        job.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
    except Exception as e:
        job.update({"status": "failed", "error": f"failed to launch ffmpeg: {e}"})
        job.done = True
        return

    def _read_err():
        for line in job.proc.stderr:
            job.log.append(line.rstrip())
            if len(job.log) > 60:
                job.log.pop(0)

    threading.Thread(target=_read_err, daemon=True).start()

    cur = {}
    for line in job.proc.stdout:
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            cur[k] = v
        if line.startswith("progress="):
            out_sec = _parse_out_time(cur)
            pct = job.last.get("progress", 0.0)
            if duration and out_sec is not None and duration > 0:
                pct = min(99.9, out_sec / duration * 100)
            job.update({
                "frame": cur.get("frame"), "fps": cur.get("fps"),
                "speed": cur.get("speed"), "outTime": out_sec, "progress": pct,
            })
            cur = {}

    rc = job.proc.wait()
    ok = rc == 0 and os.path.exists(output) and os.path.getsize(output) > 0
    if ok:
        extra = {}
        if on_success:
            try:
                extra = on_success() or {}
            except Exception as e:
                extra = {"compareError": str(e)}
        job.update({"status": "completed", "progress": 100.0,
                    "size": os.path.getsize(output), **extra})
    else:
        job.update({"status": "failed", "progress": job.last.get("progress", 0.0),
                    "error": "\n".join(job.log[-12:]) or f"ffmpeg exit {rc}"})
    job.done = True


# ---------------------------------------------------------------- upscale (TAS)


def start_upscale(input_path, model, method, factor, scale, encode, half, audio_source):
    base, _ = os.path.splitext(input_path)
    stem = base[:-8] if base.endswith("-prepped") else base
    output = stem + "-upscaled.mkv"
    job = _new_job("tas", output)
    port = _next_port()
    threading.Thread(
        target=_run_upscale,
        args=(job, input_path, output, model, method, factor, scale, encode,
              half, audio_source, port),
        daemon=True,
    ).start()
    return job


def _run_upscale(job, input_path, output, model, method, factor, scale, encode,
                 half, audio_source, port):
    cmd = [
        PYTHON, MAIN,
        "--input", input_path, "--output", output,
        "--upscale", "--upscale_method", method,
        "--custom_model", model, "--upscale_factor", str(factor),
        "--half", "True" if half else "False", "--encode_method", encode,
        "--ae", f"127.0.0.1:{port}",
    ]
    if scale:
        cmd += ["--output_scale", scale]
    # Pull audio/subs/chapters straight from the source during this encode
    # (stream copy) — no separate remux pass. ffmpegSettings.py reads this.
    env = dict(os.environ)
    if audio_source and os.path.exists(audio_source):
        env["TAS_AUDIO_SOURCE"] = audio_source
    else:
        env.pop("TAS_AUDIO_SOURCE", None)
    note = "Building TensorRT engine (first run for this model/res can take a "\
           "few minutes)…"
    if audio_source:
        note += f"  Audio/subs from {os.path.basename(audio_source)}."
    job.update({"status": "initializing", "input": input_path, "output": output,
                "port": port, "progress": 0.0, "audioSource": audio_source or None,
                "note": note})
    try:
        job.proc = subprocess.Popen(
            cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
    except Exception as e:
        job.update({"status": "failed", "error": f"failed to launch TAS: {e}"})
        job.done = True
        return

    def _read_out():
        for line in job.proc.stdout:
            job.log.append(line.rstrip())
            if len(job.log) > 200:
                job.log.pop(0)

    threading.Thread(target=_read_out, daemon=True).start()
    threading.Thread(target=_bridge_socket, args=(job, port), daemon=True).start()

    rc = job.proc.wait()
    time.sleep(1.2)  # let a terminal socket event land first
    if not job.done:
        ok = rc == 0 and os.path.exists(output) and os.path.getsize(output) > 0
        job.update({"status": "completed" if ok else "failed",
                    "progress": 100.0 if ok else job.last.get("progress", 0.0),
                    "error": None if ok else ("\n".join(job.log[-15:]) or f"exit {rc}")})
        job.done = True


def _bridge_socket(job, port):
    import socketio

    sio = socketio.Client(reconnection=True, reconnection_attempts=0)

    @sio.on("progress")
    def _on_progress(data):
        if not isinstance(data, dict):
            return
        cf = data.get("currentFrame", 0) or 0
        tf = data.get("totalFrames", 1) or 1
        pct = (cf / tf * 100) if tf else 0.0
        status = data.get("status", "processing")
        upd = {
            "status": status, "currentFrame": cf, "totalFrames": tf,
            "fps": data.get("fps"), "eta": data.get("eta"),
            "elapsed": data.get("elapsedTime"),
        }
        if status not in ("completed", "failed"):
            upd["progress"] = pct
            if cf > 0:
                upd["note"] = None
        if data.get("outputPath"):
            upd["output"] = data["outputPath"]
        if data.get("error"):
            upd["error"] = data["error"]
        if status == "completed":
            upd["progress"] = 100.0
        job.update(upd)
        if status in ("completed", "failed"):
            job.done = True
            try:
                sio.disconnect()
            except Exception:
                pass

    for _ in range(180):  # ~3 min of connect retries while engine builds
        try:
            sio.connect(f"http://127.0.0.1:{port}")
            break
        except Exception:
            if job.proc and job.proc.poll() is not None:
                return
            time.sleep(1.0)
    else:
        return
    try:
        sio.wait()
    except Exception:
        pass


# ---------------------------------------------------------------- listings


def list_videos():
    raw, prepared, upscaled = [], [], []
    if os.path.isdir(VIDEO_ROOT):
        for dp, _, fs in os.walk(VIDEO_ROOT):
            for f in fs:
                low = f.lower()
                if not low.endswith(VIDEO_EXTS):
                    continue
                full = os.path.join(dp, f)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = 0
                rel = os.path.relpath(full, VIDEO_ROOT).replace("\\", "/")
                item = {"path": full.replace("\\", "/"), "rel": rel, "size": size}
                if low.endswith("-prepped.mkv"):
                    prepared.append(item)
                elif low.endswith("-upscaled.mkv"):
                    upscaled.append(item)
                elif low.endswith("-final.mkv"):
                    continue  # finalized outputs aren't re-listed as sources
                else:
                    raw.append(item)
    for lst in (raw, prepared, upscaled):
        lst.sort(key=lambda x: x["rel"].lower())
    return raw, prepared, upscaled


def list_models():
    out = []
    if os.path.isdir(MODEL_ROOT):
        for f in sorted(os.listdir(MODEL_ROOT)):
            if f.lower().endswith(MODEL_EXTS):
                full = os.path.join(MODEL_ROOT, f)
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = 0
                out.append({"path": full.replace("\\", "/"), "name": f, "size": size})
    return out


# ---------------------------------------------------------------- HTTP handler


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._serve_index()
        if path == "/api/config":
            return self._json({
                "videoRoot": VIDEO_ROOT, "modelRoot": MODEL_ROOT,
                "methods": UPSCALE_METHODS, "encoders": ENCODE_METHODS,
            })
        if path == "/api/videos":
            raw, prepared, upscaled = list_videos()
            return self._json({"raw": raw, "prepared": prepared,
                               "upscaled": upscaled})
        if path == "/api/models":
            return self._json({"models": list_models()})
        if path == "/api/state":
            out = {}
            for kind in ("prep", "tas"):
                jid = current.get(kind)
                job = jobs.get(jid) if jid else None
                if job:
                    snap, _ = job.snapshot()
                    out[kind] = {"job_id": job.id, "output": job.output,
                                 "snapshot": snap}
                else:
                    out[kind] = None
            return self._json(out)
        if path.startswith("/api/events/"):
            return self._sse(path.rsplit("/", 1)[-1])
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        data = self._body()
        if path == "/api/verify":
            inp = data.get("input")
            if not inp or not os.path.exists(inp):
                return self._json({"error": "input not found"}, 400)
            base, _ = os.path.splitext(inp)
            output = base + "-prepped.mkv"
            if not os.path.exists(output):
                return self._json(
                    {"error": "no -prepped file exists for this source yet"}, 400)
            job = start_verify(inp, output)
            return self._json({"job_id": job.id, "output": output})
        if path == "/api/detect":
            inp = data.get("input")
            if not inp or not os.path.exists(inp):
                return self._json({"error": "input not found"}, 400)
            det = _detect_cadence(inp)
            vf, rate = _prep_plan(det["cadence"])
            det["plan"] = {"vf": vf, "rate": " ".join(rate) or "(native fps)"}
            det["source"] = _probe_meta(inp)
            return self._json(det)
        if path == "/api/prepare":
            inp = data.get("input")
            if not inp or not os.path.exists(inp):
                return self._json({"error": "input not found"}, 400)
            cadence = data.get("cadence", "auto")
            if cadence not in CADENCES:
                cadence = "auto"
            job = start_prepare(inp, cadence)
            return self._json({"job_id": job.id, "output": job.output})
        if path == "/api/upscale":
            inp = data.get("input")
            model = data.get("model")
            if not inp or not os.path.exists(inp):
                return self._json({"error": "input not found"}, 400)
            if not model or not os.path.exists(model):
                return self._json({"error": "model not found"}, 400)
            src = (data.get("audio_source") or "").strip()
            if src and not os.path.exists(src):
                return self._json({"error": "audio source not found"}, 400)
            job = start_upscale(
                inp, model,
                data.get("method", "span-directml"),
                int(data.get("factor", 2)),
                (data.get("scale") or "").strip(),
                data.get("encode", "x265_10bit"),
                bool(data.get("half", False)),
                src or None,
            )
            return self._json({"job_id": job.id, "output": job.output})
        if path == "/api/cancel":
            job = jobs.get(data.get("job_id"))
            if job and job.proc and job.proc.poll() is None:
                try:
                    job.proc.kill()
                except Exception:
                    pass
                job.update({"status": "failed", "error": "cancelled by user"})
                job.done = True
                return self._json({"ok": True})
            return self._json({"error": "no such active job"}, 404)
        return self._json({"error": "not found"}, 404)

    def _serve_index(self):
        try:
            with open(INDEX, "rb") as f:
                body = f.read()
        except OSError:
            body = b"<h1>index.html missing</h1>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, jid):
        job = jobs.get(jid)
        if not job:
            return self._json({"error": "no such job"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        last_ver = -1
        try:
            while True:
                snap, ver = job.snapshot()
                if ver != last_ver:
                    last_ver = ver
                    self.wfile.write(
                        f"data: {json.dumps(snap)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    if snap.get("status") in ("completed", "failed"):
                        break
                else:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                time.sleep(0.25)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def main():
    print("=" * 60)
    print("  TAS Web UI")
    print(f"  Videos : {VIDEO_ROOT}")
    print(f"  Models : {MODEL_ROOT}")
    print(f"  ffmpeg : {'ok' if os.path.exists(FFMPEG) else 'MISSING'}")
    print(f"  URL    : http://127.0.0.1:{PORT}")
    print("=" * 60, flush=True)
    if not os.environ.get("TAS_NO_BROWSER"):
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")
        ).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
