def analyze_and_generate(y, sr, target_dur, mode, n_versions=5, tolerance=0.25):
    total_dur = len(y) / sr
    tol = target_dur * tolerance

    beat_times   = detect_beats(y, sr)
    bounds_times = detect_boundaries(y, sr, n_segments=max(4, int(total_dur / 15)))
    rms, rms_times = compute_rms(y, sr)

    if len(beat_times) < 4:
        beat_times = np.linspace(0, total_dur, int(total_dur * 2))
    if len(rms_times) == 0:
        return []

    # 限制 beat 數量避免 OOM
    beats = beat_times[::max(1, len(beat_times) // 60)]

    beat_features = {float(bt): compute_beat_feature(y, sr, float(bt)) for bt in beats}

    results   = []
    seen      = []
    MIN_SEG   = 8.0   # 每段最少 8 秒

    # ── 遍歷所有 (cut點, rejoin點) 組合 ──
    for cut_beat in beats:
        feat_cut = beat_features.get(float(cut_beat))
        if feat_cut is None:
            continue

        for rejoin_beat in beats:
            # rejoin 不能跟 cut 太近（避免接同一個地方）
            if abs(rejoin_beat - cut_beat) < MIN_SEG:
                continue

            feat_rejoin = beat_features.get(float(rejoin_beat))
            if feat_rejoin is None:
                continue

            sim = beat_similarity(feat_cut, feat_rejoin)
            if sim < 0.3:   # 相似度太低直接跳過
                continue

            # 枚舉 A 段起點
            for start_t in [0.0] + [float(b) for b in bounds_times[:-1] if b < total_dur * 0.4]:
                part_a = cut_beat - start_t
                if part_a < MIN_SEG:
                    continue

                # 計算 B 段需要多長
                part_b_needed = target_dur - part_a
                if part_b_needed < MIN_SEG:
                    continue

                # B 段結尾
                end_t = rejoin_beat + part_b_needed
                if end_t > total_dur + 2.0:
                    continue

                # 對齊到最近 beat
                end_t = float(nearest_beat(end_t, beat_times))
                actual_dur = part_a + (end_t - rejoin_beat)

                if abs(actual_dur - target_dur) > tol:
                    continue

                # 評分
                rms_end    = get_rms_at(end_t, rms, rms_times)
                rms_mean   = float(np.mean(rms)) + 1e-6
                end_score  = max(0.0, 1.0 - rms_end / rms_mean)
                bound_score = (
                    (1.0 if any(abs(cut_beat    - b) < 1.0 for b in bounds_times) else 0.3) +
                    (1.0 if any(abs(rejoin_beat - b) < 1.0 for b in bounds_times) else 0.3)
                ) / 2.0

                if mode == "loop":
                    score = sim * 0.7 + bound_score * 0.3
                else:
                    score = sim * 0.5 + bound_score * 0.3 + end_score * 0.2

                seen.append({
                    "mode":     "loop" if mode == "loop" else "shorten",
                    "start":    float(start_t),
                    "cut":      float(cut_beat),
                    "rejoin":   float(rejoin_beat),
                    "end":      float(end_t),
                    "duration": round(actual_dur, 2),
                    "score":    float(score),
                    "sim":      float(sim),
                })

                if len(seen) >= 2000:
                    break
            if len(seen) >= 2000:
                break
        if len(seen) >= 2000:
            break

    # ── 排序去重 ──
    seen.sort(key=lambda x: -x["score"])
    deduped = []
    for c in seen:
        too_close = any(
            abs(c["cut"] - d["cut"]) < 3.0 and abs(c["rejoin"] - d["rejoin"]) < 3.0
            for d in deduped
        )
        if not too_close:
            deduped.append(c)
        if len(deduped) >= n_versions:
            break

    # ── 強制備用：找不到就取前 target_dur 秒 ──
    if len(deduped) == 0:
        end_t = float(nearest_beat(min(target_dur, total_dur), beat_times))
        seg   = apply_fade(y[:int(end_t * sr)].copy(), sr)
        return [{
            "id": 1, "mode": "shorten",
            "start": 0.0, "end": end_t,
            "duration": round(end_t, 2), "score": 0.5,
            "label": f"版本 1｜0:00 → {fmt_time(end_t)}｜直接裁切（備用）",
            "audio_b64": audio_to_b64(seg, sr),
        }]

    # ── 輸出 ──
    output = []
    for idx, c in enumerate(deduped):
        spliced   = crossfade_splice(y, sr, c["start"], c["cut"], c["rejoin"], c["end"])
        spliced   = apply_fade(spliced, sr, fade_in_sec=0.05, fade_out_sec=2.0)
        audio_b64 = audio_to_b64(spliced, sr)
        score_pct = int(c["score"] * 100)
        sim_pct   = int(c["sim"]   * 100)
        label = (
            f"版本 {idx+1}｜"
            f"{fmt_time(c['start'])}→{fmt_time(c['cut'])} ✂️ "
            f"{fmt_time(c['rejoin'])}→{fmt_time(c['end'])}｜"
            f"{c['duration']}秒｜自然度 {score_pct}%｜相似度 {sim_pct}%"
        )
        output.append({
            "id":        idx + 1,
            "mode":      c["mode"],
            "start":     c["start"],
            "end":       c["end"],
            "duration":  c["duration"],
            "score":     round(c["score"], 3),
            "label":     label,
            "audio_b64": audio_b64,
        })
    return output
