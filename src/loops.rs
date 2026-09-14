//! Loop-point discovery, ported from PyMusicLooper's `analysis.py`
//! (MIT, (c) arkrow). Chroma similarity + loudness-difference candidate
//! selection, cosine sub-sequence scoring, then duration preference.

#[derive(Clone, Debug)]
pub struct LoopPair {
    pub start_frame: usize,
    pub end_frame: usize,
    pub note_distance: f32,
    pub loudness_difference: f32,
    pub score: f32,
}

const ACCEPTABLE_NOTE_DEVIATION: f32 = 0.0875;
const ACCEPTABLE_LOUDNESS_DIFFERENCE: f32 = 0.5;

#[inline]
fn col_dot(chroma: &[f32], n_frames: usize, f1: usize, f2: usize) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for c in 0..12 {
        let base = c * n_frames;
        let x = chroma[base + f1];
        let y = chroma[base + f2];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na * nb).sqrt().max(1e-10)
}

#[inline]
fn col_l2(chroma: &[f32], n_frames: usize, f: usize) -> f32 {
    let mut s = 0.0f32;
    for c in 0..12 {
        let v = chroma[c * n_frames + f];
        s += v * v;
    }
    s.sqrt()
}

#[inline]
fn mel_dot(mel: &[f32], n_frames: usize, n_mels: usize, f1: usize, f2: usize) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for m in 0..n_mels {
        let base = m * n_frames;
        let x = mel[base + f1];
        let y = mel[base + f2];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na * nb).sqrt().max(1e-10)
}

/// Combined pitch (chroma) + timbre (log-mel) frame similarity.
#[inline]
fn frame_sim(
    chroma: &[f32],
    mel: &[f32],
    n_frames: usize,
    n_mels: usize,
    f1: usize,
    f2: usize,
) -> f32 {
    0.5 * col_dot(chroma, n_frames, f1, f2) + 0.5 * mel_dot(mel, n_frames, n_mels, f1, f2)
}

#[inline]
fn col_max(power_db: &[f32], n_frames: usize, n_bins: usize, f: usize) -> f32 {
    let mut m = f32::MIN;
    for k in 0..n_bins {
        let v = power_db[k * n_frames + f];
        if v > m {
            m = v;
        }
    }
    m
}

fn geomspace(start: f32, stop: f32, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let ratio = (stop / start).powf(1.0 / (n as f32 - 1.0));
    (0..n).map(|i| start * ratio.powi(i as i32)).collect()
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (sorted.len() - 1) as f32;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f32;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[allow(clippy::too_many_arguments)]
fn subseq_sim(
    chroma: &[f32],
    mel: &[f32],
    n_frames: usize,
    n_mels: usize,
    b1_start: usize,
    b2_start: usize,
    test_end_offset: i64,
    weights: &[f32],
) -> f32 {
    let clen = n_frames as i64;
    let test_length = test_end_offset.unsigned_abs() as usize;
    let mut b1s = b1_start as i64;
    let mut b2s = b2_start as i64;

    let b1e = if test_end_offset < 0 {
        let orig_b1 = b1s;
        let max_neg = test_end_offset.max(-b1s).max(-b2s);
        b1s += max_neg;
        b2s += max_neg;
        orig_b1
    } else {
        let e1 = (b1s + test_length as i64).min(clen);
        let e2 = (b2s + test_length as i64).min(clen);
        let mo = (e1 - b1s).min(e2 - b2s);
        b1s + mo
    };

    let max_offset = (b1e - b1s).max(0) as usize;
    let sims: Vec<f32> = (0..max_offset)
        .map(|i| {
            let f1 = (b1s + i as i64) as usize;
            let f2 = (b2s + i as i64) as usize;
            frame_sim(chroma, mel, n_frames, n_mels, f1, f2)
        })
        .collect();

    let mut sum = 0.0f32;
    let mut wsum = 0.0f32;
    for (i, w) in weights.iter().enumerate().take(test_length) {
        let v = sims.get(i).copied().unwrap_or(0.0);
        sum += v * w;
        wsum += w;
    }
    if wsum <= 0.0 {
        0.0
    } else {
        sum / wsum
    }
}

#[allow(clippy::too_many_arguments)]
fn loop_score(
    chroma: &[f32],
    mel: &[f32],
    n_frames: usize,
    n_mels: usize,
    b1: usize,
    b2: usize,
    test_duration: usize,
    weights: &[f32],
) -> f32 {
    let lookahead = subseq_sim(
        chroma,
        mel,
        n_frames,
        n_mels,
        b1,
        b2,
        test_duration as i64,
        weights,
    );
    let rev: Vec<f32> = weights.iter().rev().copied().collect();
    let lookbehind = subseq_sim(
        chroma,
        mel,
        n_frames,
        n_mels,
        b1,
        b2,
        -(test_duration as i64),
        &rev,
    );
    lookahead.max(lookbehind)
}

fn find_candidate_pairs(
    chroma: &[f32],
    n_frames: usize,
    power_db: &[f32],
    n_bins: usize,
    beats: &[usize],
    min_loop_duration: usize,
    max_loop_duration: usize,
) -> Vec<LoopPair> {
    // deviation[i] = || chroma[:, beats[i]] * ACCEPTABLE_NOTE_DEVIATION ||
    let deviation: Vec<f32> = beats
        .iter()
        .map(|&b| ACCEPTABLE_NOTE_DEVIATION * col_l2(chroma, n_frames, b))
        .collect();

    let mut pairs = Vec::new();
    let deviation_of = |frame: usize| -> f32 {
        // find index of this beat; beats are sorted so binary search
        match beats.binary_search(&frame) {
            Ok(i) => deviation[i],
            Err(_) => ACCEPTABLE_NOTE_DEVIATION,
        }
    };

    for &loop_end in beats {
        for &loop_start in beats {
            if loop_end <= loop_start {
                continue;
            }
            let loop_length = loop_end - loop_start;
            if loop_length < min_loop_duration {
                break;
            }
            if loop_length > max_loop_duration {
                continue;
            }
            let note_distance = {
                let mut s = 0.0f32;
                for c in 0..12 {
                    let base = c * n_frames;
                    let d = chroma[base + loop_end] - chroma[base + loop_start];
                    s += d * d;
                }
                s.sqrt()
            };
            if note_distance <= deviation_of(loop_end) {
                let loudness_difference = (col_max(power_db, n_frames, n_bins, loop_end)
                    - col_max(power_db, n_frames, n_bins, loop_start))
                .abs();
                if loudness_difference <= ACCEPTABLE_LOUDNESS_DIFFERENCE {
                    pairs.push(LoopPair {
                        start_frame: loop_start,
                        end_frame: loop_end,
                        note_distance,
                        loudness_difference,
                        score: 0.0,
                    });
                }
            }
        }
    }
    pairs
}

fn prune_candidates(pairs: &[LoopPair]) -> Vec<LoopPair> {
    let db: Vec<f32> = pairs.iter().map(|p| p.loudness_difference).collect();
    let nd: Vec<f32> = pairs.iter().map(|p| p.note_distance).collect();

    let eps = 1e-3f32;
    let mut db_adj: Vec<f32> = db.iter().copied().filter(|&x| x > eps).collect();
    let mut nd_adj: Vec<f32> = nd.iter().copied().filter(|&x| x > eps).collect();
    db_adj.sort_by(|a, b| a.partial_cmp(b).unwrap());
    nd_adj.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let db_threshold = if db_adj.len() > 3 {
        percentile(&db_adj, 50.0)
    } else {
        db.iter().copied().fold(0.0, f32::max)
    };
    let nd_threshold = if nd_adj.len() > 3 {
        percentile(&nd_adj, 75.0)
    } else {
        nd.iter().copied().fold(0.0, f32::max)
    };
    let db_lim = db_threshold.max(0.25);

    pairs
        .iter()
        .filter(|p| p.loudness_difference <= db_lim && p.note_distance <= nd_threshold)
        .cloned()
        .collect()
}

fn prioritize_duration(pairs: &mut Vec<LoopPair>) {
    if pairs.len() < 2 {
        return;
    }
    let mut db: Vec<f32> = pairs.iter().map(|p| p.loudness_difference).collect();
    db.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let db_threshold = percentile(&db, 50.0);

    let mut scores: Vec<f32> = pairs.iter().map(|p| p.score).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut score_threshold = percentile(&scores, 90.0);
    score_threshold = score_threshold.max(pairs[0].score - 1e-4);

    let mut duration_argmax = 0usize;
    let mut duration_max = 0usize;
    for (idx, p) in pairs.iter().enumerate() {
        if p.score < score_threshold {
            break;
        }
        let duration = p.end_frame - p.start_frame;
        if duration > duration_max && p.loudness_difference <= db_threshold {
            duration_max = duration;
            duration_argmax = idx;
        }
    }
    if duration_argmax != 0 {
        let p = pairs.remove(duration_argmax);
        pairs.insert(0, p);
    }
}

/// Find and rank loop points. Returns a score-sorted list (best first).
#[allow(clippy::too_many_arguments)]
pub fn find_best_loop_points(
    chroma: &[f32],
    mel: &[f32],
    n_mels: usize,
    power_db: &[f32],
    n_bins: usize,
    n_frames: usize,
    beats: &[usize],
    bpm: f32,
    sample_rate: u32,
    min_loop_frames: usize,
    max_loop_frames: usize,
    disable_pruning: bool,
) -> Vec<LoopPair> {
    if beats.len() < 2 {
        return Vec::new();
    }
    let min_loop_frames = min_loop_frames.max(1);
    let max_loop_frames = max_loop_frames.max(min_loop_frames + 1);

    let mut pairs = find_candidate_pairs(
        chroma,
        n_frames,
        power_db,
        n_bins,
        beats,
        min_loop_frames,
        max_loop_frames,
    );
    if pairs.is_empty() {
        return pairs;
    }

    let bpm = if bpm > 1.0 { bpm } else { 120.0 };
    let beats_per_second = bpm / 60.0;
    let seconds_to_test = 12.0 / beats_per_second;
    let mut test_offset = (seconds_to_test * sample_rate as f32 / crate::dsp::HOP as f32)
        .round()
        .max(1.0) as usize;
    if test_offset > n_frames {
        test_offset = n_frames / 4;
    }
    test_offset = test_offset.max(1);
    let weights = geomspace((test_offset / 12).max(2) as f32, 1.0, test_offset);

    if pairs.len() >= 100 && !disable_pruning {
        pairs = prune_candidates(&pairs);
        if pairs.is_empty() {
            return pairs;
        }
    }

    for p in pairs.iter_mut() {
        p.score = loop_score(
            chroma,
            mel,
            n_frames,
            n_mels,
            p.start_frame,
            p.end_frame,
            test_offset,
            &weights,
        );
    }
    pairs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    prioritize_duration(&mut pairs);
    pairs
}

/// Audacity-style "at zero crossings" snap (rising crossing, lowest local energy).
pub fn nearest_zero_crossing(mono: &[f32], sample_rate: u32, sample_idx: usize) -> usize {
    let n = mono.len();
    if n == 0 {
        return sample_idx;
    }
    let sample_idx = sample_idx.min(n - 1);
    let window_size = (sample_rate as f32 / 100.0).max(1.0) as usize;
    let offset = window_size / 2;
    let neg = sample_idx.saturating_sub(offset);
    let pos = (sample_idx + offset).min(n);
    let window = &mono[neg..pos];
    if window.is_empty() {
        return sample_idx;
    }
    let offset_correction = offset.saturating_sub(sample_idx);

    let mut dist = vec![0.0f32; window.len()];
    let mut prev = 2.0f32;
    for (i, &x) in window.iter().enumerate() {
        let mut fdist = x.abs();
        if prev * x > 0.0 {
            fdist += 0.4;
        } else if prev > 0.0 {
            fdist += 0.1;
        }
        prev = x;
        dist[i] = fdist;
    }
    for (i, d) in dist.iter_mut().enumerate() {
        *d += 0.1 * (i as f32 - offset as f32 + offset_correction as f32).abs()
            / (window_size as f32 / 2.0);
    }

    let mut argmin = 0usize;
    let mut minv = f32::MAX;
    for (i, &d) in dist.iter().enumerate() {
        if d < minv {
            minv = d;
            argmin = i;
        }
    }
    if minv > 0.2 {
        return sample_idx;
    }
    let res = sample_idx as isize + argmin as isize - offset as isize + offset_correction as isize;
    res.clamp(0, n as isize - 1) as usize
}
