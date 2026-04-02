from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import soundfile as sf
import io, base64, tempfile, os

app = Flask(__name__)
CORS(app)

# ── 工具：秒數格式化 ──
def fmt_time(sec):
    m = int(sec // 60)
    s = sec % 60
    return f"{m}:{s:05.2f}"

# ── 工具：找最近的 beat ──
def nearest_beat(t, beats):
    idx = np.argmin(np.abs(beats - t))
    return beats[idx]

# ── 工具：音訊片段轉 base64 ──
def audio_to_b64(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='mp3', subtype='PCM_16')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

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

# ══════════════════════════════════════════
#  核心分析函式
# ══════════════════════════════════════════
def analyze_and_generate(y, sr, target_dur, mode, n_versions=5, tolerance=0.15):
    total_dur = librosa.get_duration(y=y, sr=sr)

    # 1. 節拍偵測
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # 2. 段落邊界偵測（onset strength + 自訂分段）
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    bounds_frames = librosa.segment.agglomerative(onset_env, k=max(4, int(total_dur / 15)))
    bounds_times  = librosa.frames_to_time(bounds_frames, sr=sr)
    # 加入開頭與結尾
    bounds_times = np.unique(np.concatenate([[0.0], bounds_times, [total_dur]]))

    # 3. RMS 能量（用於音量平滑分數）
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    def get_rms_at(t):
        idx = np.argmin(np.abs(rms_times - t))
        return float(rms[idx])

    # 4. 候選剪接方案
    candidates = []
    tol = target_dur * tolerance  # 允許誤差

    if mode in ('shorten', 'both'):
        # 從段落邊界組合 (start, end)
        for i, s in enumerate(bounds_times[:-1]):
            for e in bounds_times[i+1:]:
                dur = e - s
                if abs(dur - target_dur) <= tol:
                    # snap 到最近 beat
                    s_beat = nearest_beat(s, beat_times)
                    e_beat = nearest_beat(e, beat_times)
                    if e_beat <= s_beat:
                        continue
                    actual_dur = e_beat - s_beat
                    if abs(actual_dur - target_dur) > tol * 1.5:
                        continue

                    # 計算自然度分數
                    beat_align = 1.0 - (abs(s - s_beat) + abs(e - e_beat)) / (target_dur + 1e-6)
                    bound_score = (
                        (1.0 if s in bounds_times else 0.5) +
                        (1.0 if e in bounds_times else 0.5)
                    ) / 2.0
                    rms_s = get_rms_at(s_beat)
                    rms_e = get_rms_at(e_beat)
                    rms_mean = np.mean(rms) + 1e-6
                    smooth_score = 1.0 - abs(rms_s - rms_e) / rms_mean
                    smooth_score = max(0.0, smooth_score)

                    score = beat_align * 0.4 + bound_score * 0.4 + smooth_score * 0.2

                    candidates.append({
                        'mode': 'shorten',
                        'start': float(s_beat),
                        'end':   float(e_beat),
                        'duration': float(e_beat - s_beat),
                        'score': float(score),
                    })

    if mode in ('loop', 'both'):
        # 找無縫循環：用 chroma 相似度找頭尾相近的區間
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr)

        def chroma_at(t):
            idx = np.argmin(np.abs(chroma_times - t))
            return chroma[:, idx]

        for i, s in enumerate(bounds_times[:-1]):
            for e in bounds_times[i+1:]:
                dur = e - s
                if abs(dur - target_dur) > tol:
                    continue
                s_beat = nearest_beat(s, beat_times)
                e_beat = nearest_beat(e, beat_times)
                if e_beat <= s_beat:
                    continue

                # chroma 相似度（頭尾音色相近 → 循環更自然）
                c_start = chroma_at(s_beat)
                c_end   = chroma_at(e_beat)
                sim = float(np.dot(c_start, c_end) / (
                    np.linalg.norm(c_start) * np.linalg.norm(c_end) + 1e-6
                ))

                beat_align = 1.0 - (abs(s - s_beat) + abs(e - e_beat)) / (target_dur + 1e-6)
                score = sim * 0.6 + beat_align * 0.4

                candidates.append({
                    'mode': 'loop',
                    'start': float(s_beat),
                    'end':   float(e_beat),
                    'duration': float(e_beat - s_beat),
                    'score': float(score),
                })

    # 5. 排序、去重、取前 N
    candidates.sort(key=lambda x: -x['score'])

    # 去重（避免 start 太接近的版本）
    deduped = []
    for c in candidates:
        too_close = any(abs(c['start'] - d['start']) < 2.0 for d in deduped)
        if not too_close:
            deduped.append(c)
        if len(deduped) >= n_versions:
            break

    # 6. 剪裁音訊 + 轉 base64
    results = []
    for idx, c in enumerate(deduped):
        s_sample = int(c['start'] * sr)
        e_sample = int(c['end']   * sr)
        segment  = y[s_sample:e_sample]
        segment  = apply_fade(segment, sr)
        audio_b64 = audio_to_b64(segment, sr)

        score_pct = int(c['score'] * 100)
        label = f"版本 {idx+1}｜{fmt_time(c['start'])} → {fmt_time(c['end'])}｜自然度 {score_pct}%"
        if c['mode'] == 'loop':
            label += "｜🔁 循環"

        results.append({
            'id':       idx + 1,
            'mode':     c['mode'],
            'start':    round(c['start'], 2),
            'end':      round(c['end'],   2),
            'duration': round(c['duration'], 2),
            'score':    round(c['score'], 3),
            'label':    label,
            'audio_b64': audio_b64,
        })

    return results

# ══════════════════════════════════════════
#  路由
# ══════════════════════════════════════════

@app.route('/')
def index():
    return jsonify({'status': 'AudioCraft API running'})

@app.route('/analyze-and-cut', methods=['POST'])
def analyze_and_cut():
    if 'file' not in request.files:
        return jsonify({'error': '請上傳音訊檔案'}), 400

    file = request.files['file']
    target_dur = float(request.form.get('target_duration', 30))
    mode       = request.form.get('mode', 'both')  # shorten / loop / both
    n_versions = int(request.form.get('n_versions', 5))

    if target_dur <= 0:
        return jsonify({'error': '目標時長必須大於 0'}), 400

    # 儲存暫存檔
    suffix = os.path.splitext(file.filename)[-1] or '.mp3'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
        total_dur = librosa.get_duration(y=y, sr=sr)

        if target_dur >= total_dur:
            return jsonify({'error': f'目標時長（{target_dur}s）不能大於或等於原始時長（{total_dur:.1f}s）'}), 400

        versions = analyze_and_generate(y, sr, target_dur, mode, n_versions)

        if not versions:
            return jsonify({'error': '找不到合適的剪接點，請嘗試調整目標時長'}), 422

        return jsonify({
            'total_duration': round(total_dur, 2),
            'target_duration': target_dur,
            'mode': mode,
            'versions': versions,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)

# 保留舊路由（相容性）
@app.route('/convert', methods=['POST'])
def convert():
    if 'file' not in request.files:
        return jsonify({'error': '請上傳檔案'}), 400
    file   = request.files['file']
    fmt    = request.form.get('format', 'mp3')
    suffix = os.path.splitext(file.filename)[-1] or '.mp3'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=False)
        if y.ndim > 1:
            y = y.T
        buf = io.BytesIO()
        sf.write(buf, y, sr, format=fmt)
        buf.seek(0)
        from flask import send_file
        return send_file(buf, mimetype=f'audio/{fmt}',
                         as_attachment=True, download_name=f'output.{fmt}')
    finally:
        os.unlink(tmp_path)

@app.route('/trim', methods=['POST'])
def trim():
    if 'file' not in request.files:
        return jsonify({'error': '請上傳檔案'}), 400
    file  = request.files['file']
    start = float(request.form.get('start', 0))
    end   = float(request.form.get('end', 0))
    suffix = os.path.splitext(file.filename)[-1] or '.mp3'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True, offset=start,
                             duration=(end - start) if end > start else None)
        buf = io.BytesIO()
        sf.write(buf, y, sr, format='mp3')
        buf.seek(0)
        from flask import send_file
        return send_file(buf, mimetype='audio/mp3',
                         as_attachment=True, download_name='trimmed.mp3')
    finally:
        os.unlink(tmp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
