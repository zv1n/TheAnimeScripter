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
CADENCES = ("auto", "film", "progressive", "interlaced")


def _prep_vf(cadence):
    """Video-filter chain for a given cadence."""
    if cadence == "film":       # 3:2-telecined film → inverse telecine to 23.976
        return f"fieldmatch,yadif=deint=interlaced,decimate,{_SCALE}"
    if cadence == "interlaced":  # true interlaced video → deinterlace, keep fps
        return f"yadif,{_SCALE}"
    return _SCALE                # progressive → no field/rate changes


def _detect_cadence(path):
    """Sniff cadence with idet on a mid-file sample. Returns film/progressive/
    interlaced. Repeated fields ⇒ telecine; else interlaced-heavy ⇒ interlaced."""
    dur = _probe_duration(path) or 0
    ss = max(0, dur * 0.4)
    cmd = [FFMPEG, "-hide_banner", "-ss", str(ss), "-i", path,
           "-vf", "idet", "-frames:v", "500", "-an", "-f", "null", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180).stderr
    except Exception:
        return "progressive"

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
    total = max(1, tff + bff + prog)
    if repeated / total > 0.05:      # 3:2 pulldown leaves ~1/5 repeated fields
        return "film"
    if (tff + bff) / total > 0.10:
        return "interlaced"
    return "progressive"


def start_prepare(input_path, cadence="auto"):
    base, _ = os.path.splitext(input_path)
    output = base + "-prepped.mkv"
    job = _new_job("prep", output)
    threading.Thread(
        target=_run_prepare, args=(job, input_path, output, cadence), daemon=True
    ).start()
    return job


def _run_prepare(job, input_path, output, cadence):
    duration = _probe_duration(input_path)
    if cadence == "auto":
        job.update({"status": "detecting", "input": input_path, "output": output,
                    "totalDuration": duration, "progress": 0.0,
                    "note": "Detecting cadence (idet)…"})
        cadence = _detect_cadence(input_path)
    job.update({"status": "processing", "input": input_path, "output": output,
                "totalDuration": duration, "progress": 0.0, "cadence": cadence,
                "note": None})
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-i", input_path,
        "-map", "0:v:0", "-vf", _prep_vf(cadence),
        "-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1",
        "-progress", "pipe:1", "-nostats", output,
    ]
    _run_ffmpeg_with_progress(job, cmd, output, duration)


def _run_ffmpeg_with_progress(job, cmd, output, duration):
    """Run an ffmpeg command, streaming -progress into the job snapshot."""
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
        job.update({"status": "completed", "progress": 100.0,
                    "size": os.path.getsize(output)})
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
