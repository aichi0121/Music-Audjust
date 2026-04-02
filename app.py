from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import soundfile as sf
import io, base64, tempfile, os
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

# ── 工具：秒數格式化 ──
def fmt_time(sec):
    m = int(sec // 60)
    s = sec % 60
    return f"{m}:{s:05.2f}"

# ── 工具：讀取音訊（統一轉 mono, sr=22050）──
def load_audio(path, sr=22050):
    audio = AudioSegment.from_file(path)
    audio = audio.set_frame_rate(sr).set_channels(1)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= np.iinfo(audio.array_type).max
    return samples, sr

# ── 工具：簡易節拍偵測（能量峰值法）──
def detect_beats(y, sr, hop=512):
    frame_size = hop
    n_frames = len(y) // frame_size
    energy = np.array([
        np.sum(y[i*frame_size:(i+1)*frame_size]**2)
        for i in range(n_frames)
    ])
    energy_smooth = uniform_filter1d(energy, size=5)
    peaks, _ = find_peaks(energy_smooth, distance=sr//(hop*2))
    beat_times = peaks * frame_size / sr
    return beat_times

# ── 工具：找最近的 beat ──
def nearest_beat(t, beats):
    if len(beats) == 0:
        return t
    idx = np.argmin(np.abs(beats - t))
    return beats[idx]

# ── 工具：段落邊界偵測（能量變化法）──
def detect_boundaries(y, sr, n_segments=8):
    hop = 512
    frame_size = hop
    n_frames = len(y) // frame_size
    energy = np.array([
        np.sum(y[i*frame_size:(i+1)*frame_size]**2)
        for i in range(n_frames)
    ])
    diff = np.abs(np.diff(uniform_filter1d(energy, size=10)))
    peaks, _ = find_peaks(diff, distance=n_frames // (n_segments * 2))
    if len(peaks) > n_segments:
        top = np.argsort(diff[peaks])[-n_segments:]
        peaks = np.sort(peaks[top])
    bound_times = peaks * frame_size / sr
    total_dur = len(y) / sr
    bound_times = np.unique(np.concatenate([[0.0], bound_times, [total_dur]]))
    return bound_times

# ── 工具：音訊片段轉 base64 ──
def audio_to_b64(y, sr):
    y_int16 = (y * 32767).astype(np.int16)
    audio_seg = AudioSegment(
        y_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    buf = io.BytesIO()
    audio_seg.export(buf, format="mp3")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ── 工具：加淡入淡出 ──
def apply_fade(y, sr, fade_sec=0.08):
    fade_len = int(sr * fade_sec)
    fade_len = min(fade_len, len(y) // 4)
    fade_in  = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    y = y.copy()
    y[:fade_len]  *= fade_in
    y[-fade_len:] *= fade_out
    return y

# ── RMS 能量 ──
def compute_rms(y, sr, hop=512):
    frame_size = hop
    n_frames = len(y) // frame_size
    rms = np.array([
        np.sqrt(np.mean(y[i*frame_size:(i+1)*frame_size]**2) + 1e-10)
        for i in range(n_frames)
    ])
    rms_times = np.arange(n_frames) * frame_size / sr
    return rms, rms_times

def get_rms_at(t, rms, rms_times):
    idx = np.argmin(np.abs(rms_times - t))
    return float(rms[idx])

# ══════════════════════════════════════════
#  核心分析函式
# ══════════════════════════════════════════
def analyze_and_generate(y, sr, target_dur, mode, n_versions=5, tolerance=0.15):
    total_dur = len(y) / sr
    tol = target_dur * tolerance
    beat_times     = detect_beats(y, sr)
    bounds_times   = detect_boundaries(y, sr, n_segments=max(4, int(total_dur / 15)))
    rms, rms_times = compute_rms(y, sr)
    candidates = []

    for i, s in enumerate(bounds_times[:-1]):
        for e in bounds_times[i+1:]:
            dur = e - s
            if abs(dur - target_dur) > tol:
                continue
            s_beat = nearest_beat(s, beat_times)
            e_beat = nearest_beat(e, beat_times)
            if e_beat <= s_beat:
                continue
            actual_dur = e_beat - s_beat
            if abs(actual_dur - target_dur) > tol * 1.5:
                continue
            beat_align = 1.0 - (abs(s - s_beat) + abs(e - e_beat)) / (target_dur + 1e-6)
            bound_score = (
                (1.0 if s in bounds_times else 0.5) +
                (1.0 if e in bounds_times else 0.5)
            ) / 2.0
            rms_s = get_rms_at(s_beat, rms, rms_times)
            rms_e = get_rms_at(e_beat, rms, rms_times)
            rms_mean = np.mean(rms) + 1e-6
            smooth_score = max(0.0, 1.0 - abs(rms_s - rms_e) / rms_mean)
            if mode == "loop":
                score  = smooth_score * 0.6 + beat_align * 0.4
                c_mode = "loop"
            else:
                score  = beat_align * 0.4 + bound_score * 0.4 + smooth_score * 0.2
                c_mode = "shorten"
            candidates.append({
                "mode":     c_mode,
                "start":    float(s_beat),
                "end":      float(e_beat),
                "duration": float(e_beat - s_beat),
                "score":    float(score),
            })

    candidates.sort(key=lambda x: -x["score"])
    deduped = []
    for c in candidates:
        too_close = any(abs(c["start"] - d["start"]) < 2.0 for d in deduped)
        if not too_close:
            deduped.append(c)
        if len(deduped) >= n_versions:
            break

    results = []
    for idx, c in enumerate(deduped):
        s_sample  = int(c["start"] * sr)
        e_sample  = int(c["end"]   * sr)
        segment   = y[s_sample:e_sample]
        segment   = apply_fade(segment, sr)
        audio_b64 = audio_to_b64(segment, sr)
        score_pct = int(c["score"] * 100)
        label = f"版本 {idx+1}｜{fmt_time(c['start'])} → {fmt_time(c['end'])}｜自然度 {score_pct}%"
        if c["mode"] == "loop":
            label += "｜🔁 循環"
        results.append({
            "id":        idx + 1,
            "mode":      c["mode"],
            "start":     round(c["start"], 2),
            "end":       round(c["end"],   2),
            "duration":  round(c["duration"], 2),
            "score":     round(c["score"], 3),
            "label":     label,
            "audio_b64": audio_b64,
        })
    return results

# ══════════════════════════════════════════
#  路由
# ══════════════════════════════════════════

@app.route("/")
def index():
    return jsonify({"status": "AudioCraft API running"})

@app.route("/analyze-and-cut", methods=["POST"])
def analyze_and_cut():
    if "file" not in request.files:
        return jsonify({"error": "請上傳音訊檔案"}), 400
    file       = request.files["file"]
    target_dur = float(request.form.get("target_duration", 30))
    mode       = request.form.get("mode", "both")
    n_versions = int(request.form.get("n_versions", 5))
    if target_dur <= 0:
        return jsonify({"error": "目標時長必須大於 0"}), 400
    suffix = os.path.splitext(file.filename)[-1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        y, sr     = load_audio(tmp_path)
        total_dur = len(y) / sr
        if target_dur >= total_dur:
            return jsonify({"error": f"目標時長（{target_dur}s）不能大於或等於原始時長（{total_dur:.1f}s）"}), 400
        versions = analyze_and_generate(y, sr, target_dur, mode, n_versions)
        if not versions:
            return jsonify({"error": "找不到合適的剪接點，請嘗試調整目標時長"}), 422
        return jsonify({
            "total_duration":  round(total_dur, 2),
            "target_duration": target_dur,
            "mode":            mode,
            "versions":        versions,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)

@app.route("/convert", methods=["POST"])
def convert():
    if "file" not in request.files:
        return jsonify({"error": "請上傳檔案"}), 400
    file   = request.files["file"]
    fmt    = request.form.get("format", "mp3")
    suffix = os.path.splitext(file.filename)[-1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        audio = AudioSegment.from_file(tmp_path)
        buf   = io.BytesIO()
        audio.export(buf, format=fmt)
        buf.seek(0)
        return send_file(buf, mimetype=f"audio/{fmt}",
                         as_attachment=True, download_name=f"output.{fmt}")
    finally:
        os.unlink(tmp_path)

@app.route("/trim", methods=["POST"])
def trim():
    if "file" not in request.files:
        return jsonify({"error": "請上傳檔案"}), 400
    file   = request.files["file"]
    start  = float(request.form.get("start", 0))
    end    = float(request.form.get("end", 0))
    suffix = os.path.splitext(file.filename)[-1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        audio    = AudioSegment.from_file(tmp_path)
        start_ms = int(start * 1000)
        end_ms   = int(end   * 1000) if end > start else len(audio)
        trimmed  = audio[start_ms:end_ms]
        buf = io.BytesIO()
        trimmed.export(buf, format="mp3")
        buf.seek(0)
        return send_file(buf, mimetype="audio/mp3",
                         as_attachment=True, download_name="trimmed.mp3")
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
