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
def apply_fade(y, sr, fade_in_sec=0.05, fade_out_sec=2.0):
    y = y.copy()
    fade_in_len  = min(int(sr * fade_in_sec),  len(y) // 4)
    fade_out_len = min(int(sr * fade_out_sec), len(y) // 3)
    y[:fade_in_len]   *= np.linspace(0, 1, fade_in_len)
    y[-fade_out_len:] *= np.linspace(1, 0, fade_out_len)
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
#  新增 helper 函式（只給 analyze_and_generate 用）
# ══════════════════════════════════════════

def find_good_start_points(y, sr, beat_times, bounds_times, rms, rms_times, max_start_sec=30):
    """
    找「感覺像開頭」的候選起始點：
    1. 第0秒（一定包含）
    2. 段落邊界點（bounds_times，排除最後一個）
    3. 能量從低→高的上升點
    全部對齊到最近的 beat，並限制在 max_start_sec 以內
    """
    candidates = set()
    candidates.add(0.0)

    for b in bounds_times[:-1]:
        candidates.add(b)

    hop = 512
    frame_size = hop
    n_frames = len(y) // frame_size
    energy = np.array([
        np.sum(y[i*frame_size:(i+1)*frame_size]**2)
        for i in range(n_frames)
    ])
    energy_smooth = uniform_filter1d(energy, size=20)
    mean_e = np.mean(energy_smooth)
    for i in range(10, n_frames - 10):
        before = np.mean(energy_smooth[max(0, i-10):i])
        after  = np.mean(energy_smooth[i:i+10])
        t = i * frame_size / sr
        if after > before * 1.5 and after > mean_e * 0.5 and t <= max_start_sec:
            candidates.add(round(t, 3))

    snapped = set()
    for t in candidates:
        if t > max_start_sec:
            continue
        bt = nearest_beat(t, beat_times)
        snapped.add(round(bt, 3))

    return sorted(snapped)


def compute_beat_feature(y, sr, t, hop=512, window_sec=0.5):
    """
    計算某個時間點附近的音訊特徵（RMS + 簡易頻譜重心）
    """
    half    = int(sr * window_sec / 2)
    center  = int(t * sr)
    start   = max(0, center - half)
    end     = min(len(y), center + half)
    segment = y[start:end]
    if len(segment) == 0:
        return np.array([0.0, 0.0])

    rms = np.sqrt(np.mean(segment**2) + 1e-10)
    fft_mag = np.abs(np.fft.rfft(segment))
    freqs   = np.fft.rfftfreq(len(segment), d=1.0/sr)
    spectral_centroid      = np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-10)
    spectral_centroid_norm = spectral_centroid / (sr / 2)
    return np.array([rms, spectral_centroid_norm])


def beat_similarity(feat_a, feat_b):
    diff    = np.abs(feat_a - feat_b)
    rms_sim = max(0.0, 1.0 - diff[0] / 0.1)
    sc_sim  = max(0.0, 1.0 - diff[1] / 0.2)
    return rms_sim * 0.5 + sc_sim * 0.5


def crossfade_splice(y, sr, start_time, cut_time, rejoin_time, end_time, fade_ms=300):
    """
    拼接兩段音訊並做 crossfade
    段落A：y[start_time → cut_time]
    段落B：y[rejoin_time → end_time]
    """
    fade_len = int(sr * fade_ms / 1000)
    s_a = int(start_time  * sr)
    e_a = int(cut_time    * sr)
    s_b = int(rejoin_time * sr)
    e_b = int(end_time    * sr)

    seg_a = y[s_a:e_a].copy()
    seg_b = y[s_b:e_b].copy()

    if len(seg_a) == 0 or len(seg_b) == 0:
        return np.concatenate([seg_a, seg_b])

    fade_len = min(fade_len, len(seg_a) // 4, len(seg_b) // 4)
    if fade_len < 2:
        return np.concatenate([seg_a, seg_b])

    seg_a[-fade_len:] *= np.linspace(1, 0, fade_len)
    seg_b[:fade_len]  *= np.linspace(0, 1, fade_len)
    return np.concatenate([seg_a, seg_b])


def find_natural_ending(beat_times, rms, rms_times, from_time, total_dur):
    """
    從 from_time 之後，找能量自然下降的結尾點
    """
    candidates = beat_times[beat_times > from_time]
    if len(candidates) == 0:
        return total_dur

    mean_rms = np.mean(rms)
    low_energy_beats = []
    for bt in candidates:
        r = get_rms_at(bt, rms, rms_times)
        if r < mean_rms * 0.8:
            low_energy_beats.append(bt)

    if low_energy_beats:
        for bt in reversed(low_energy_beats):
            if bt < total_dur - 1.0:
                return bt

    for bt in reversed(candidates):
        if bt < total_dur - 1.0:
            return bt

    return total_dur


# ══════════════════════════════════════════
#  核心分析函式（新版）
# ══════════════════════════════════════════
def analyze_and_generate(y, sr, target_dur, mode, n_versions=5, tolerance=0.15):
    total_dur  = len(y) / sr
    tol        = target_dur * tolerance

    beat_times   = detect_beats(y, sr)
    bounds_times = detect_boundaries(y, sr, n_segments=max(4, int(total_dur / 15)))
    rms, rms_times = compute_rms(y, sr)

    if len(beat_times) < 4:
        return []

    beat_features = {bt: compute_beat_feature(y, sr, bt) for bt in beat_times}

    start_points = find_good_start_points(
        y, sr, beat_times, bounds_times, rms, rms_times,
        max_start_sec=min(30, total_dur * 0.3)
    )

    candidates = []

    for start_t in start_points:
        beats_after_start = beat_times[beat_times > start_t + 5.0]

        for cut_beat in beats_after_start:
            feat_x = beat_features[cut_beat]
            beats_for_rejoin = beat_times[beat_times > cut_beat + 5.0]

            for rejoin_beat in beats_for_rejoin:
                end_t = find_natural_ending(
                    beat_times, rms, rms_times,
                    from_time=rejoin_beat + 2.0,
                    total_dur=total_dur
                )

                part_a = cut_beat - start_t
                part_b = end_t    - rejoin_beat
                total  = part_a + part_b

                if abs(total - target_dur) > tol:
                    continue

                feat_y      = beat_features[rejoin_beat]
                sim         = beat_similarity(feat_x, feat_y)
                bound_score = (
                    (1.0 if any(abs(cut_beat   - b) < 0.5 for b in bounds_times) else 0.3) +
                    (1.0 if any(abs(rejoin_beat - b) < 0.5 for b in bounds_times) else 0.3)
                ) / 2.0
                rms_end   = get_rms_at(end_t, rms, rms_times)
                rms_mean  = np.mean(rms) + 1e-6
                end_score = max(0.0, 1.0 - rms_end / rms_mean)

                if mode == "loop":
                    score = sim * 0.7 + bound_score * 0.3
                else:
                    score = sim * 0.5 + bound_score * 0.3 + end_score * 0.2

                candidates.append({
                    "mode":       "loop" if mode == "loop" else "shorten",
                    "start":      float(start_t),
                    "cut":        float(cut_beat),
                    "rejoin":     float(rejoin_beat),
                    "end":        float(end_t),
                    "duration":   round(total, 2),
                    "score":      float(score),
                    "similarity": float(sim),
                })

    candidates.sort(key=lambda x: -x["score"])
    deduped = []
    for c in candidates:
        too_close = any(
            abs(c["start"] - d["start"]) < 2.0 and abs(c["cut"] - d["cut"]) < 2.0
            for d in deduped
        )
        if not too_close:
            deduped.append(c)
        if len(deduped) >= n_versions:
            break

    results = []
    for idx, c in enumerate(deduped):
        spliced   = crossfade_splice(y, sr, c["start"], c["cut"], c["rejoin"], c["end"])
        spliced   = apply_fade(spliced, sr, fade_in_sec=0.05, fade_out_sec=2.0)
        audio_b64 = audio_to_b64(spliced, sr)
        score_pct = int(c["score"] * 100)
        sim_pct   = int(c["similarity"] * 100)

        label = (
            f"版本 {idx+1}｜"
            f"{fmt_time(c['start'])} → {fmt_time(c['cut'])} ✂️ "
            f"{fmt_time(c['rejoin'])} → {fmt_time(c['end'])}｜"
            f"自然度 {score_pct}%｜相似度 {sim_pct}%"
        )
        if c["mode"] == "loop":
            label += "｜🔁 循環"

        results.append({
            "id":        idx + 1,
            "mode":      c["mode"],
            "start":     round(c["start"],  2),
            "end":       round(c["end"],    2),
            "duration":  round(c["duration"], 2),
            "score":     round(c["score"],  3),
            "label":     label,
            "audio_b64": audio_b64,
        })

    return results


# ══════════════════════════════════════════
#  路由（完全不動）
# ══════════════════════════════════════════
@app.route("/analyze-and-cut", methods=["POST"])
def analyze_and_cut():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file       = request.files["file"]
    target_dur = float(request.form.get("target_duration", 30))
    mode       = request.form.get("mode", "shorten")
    n_versions = int(request.form.get("n_versions", 5))

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        y, sr   = load_audio(tmp_path)
        results = analyze_and_generate(y, sr, target_dur, mode, n_versions)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/convert", methods=["POST"])
def convert():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file   = request.files["file"]
    fmt    = request.form.get("format", "mp3")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        audio = AudioSegment.from_file(tmp_path)
        buf   = io.BytesIO()
        audio.export(buf, format=fmt)
        buf.seek(0)
        return send_file(buf, mimetype=f"audio/{fmt}", as_attachment=True,
                         download_name=f"converted.{fmt}")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/trim", methods=["POST"])
def trim():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file      = request.files["file"]
    start_sec = float(request.form.get("start", 0))
    end_sec   = float(request.form.get("end",   30))

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        audio    = AudioSegment.from_file(tmp_path)
        trimmed  = audio[int(start_sec * 1000):int(end_sec * 1000)]
        buf      = io.BytesIO()
        trimmed.export(buf, format="mp3")
        buf.seek(0)
        return send_file(buf, mimetype="audio/mp3", as_attachment=True,
                         download_name="trimmed.mp3")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True)
