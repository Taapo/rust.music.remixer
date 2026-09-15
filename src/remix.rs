//! Rearranging remix planner.
//!
//! Builds a bar-level similarity graph and finds a minimum-cost path that
//! traverses the song from its intro to its outro in the number of bars needed
//! to hit the target length. Shorter targets force forward skips, longer targets
//! force loops, and the cost model makes the path jump only where one section is
//! a natural continuation of another (the "Infinite Jukebox" / Adobe Remix
//! approach).

use crate::assemble::PlanSeg;

/// (similarity floor, neighbour count) attempts, from strict to relaxed.
const ATTEMPTS: [(f32, usize); 3] = [(0.70, 12), (0.55, 16), (0.40, 24)];
const JUMP_PENALTY: f32 = 0.05;
/// Extra cost proportional to the *square* of jump length (fraction of the
/// song), so compression is spread across the song rather than dumped into one
/// giant skip.
const DIST_PENALTY: f32 = 2.0;

struct Bars {
    starts: Vec<usize>,
    ends: Vec<usize>,
    chroma: Vec<Vec<f32>>,
    mel: Vec<Vec<f32>>,
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na * nb).sqrt().max(1e-10)
}

fn build_bars(
    chroma: &[f32],
    mel: &[f32],
    n_frames: usize,
    n_mels: usize,
    boundaries: &[usize],
) -> Bars {
    let mut starts: Vec<usize> = vec![0];
    for &b in boundaries {
        if b > 0 && b < n_frames && *starts.last().unwrap() < b {
            starts.push(b);
        }
    }

    let mut bars = Bars {
        starts: Vec::new(),
        ends: Vec::new(),
        chroma: Vec::new(),
        mel: Vec::new(),
    };
    for i in 0..starts.len() {
        let s = starts[i];
        let e = if i + 1 < starts.len() { starts[i + 1] } else { n_frames };
        if e <= s || (e - s) < 4 {
            continue;
        }
        let count = (e - s) as f32;
        let mut cb = vec![0.0f32; 12];
        let mut mb = vec![0.0f32; n_mels];
        for f in s..e {
            for c in 0..12 {
                cb[c] += chroma[c * n_frames + f];
            }
            for m in 0..n_mels {
                mb[m] += mel[m * n_frames + f];
            }
        }
        for v in cb.iter_mut() {
            *v /= count;
        }
        for v in mb.iter_mut() {
            *v /= count;
        }
        bars.starts.push(s);
        bars.ends.push(e);
        bars.chroma.push(cb);
        bars.mel.push(mb);
    }
    bars
}

fn bar_sim(bars: &Bars, i: usize, j: usize) -> f32 {
    0.5 * cosine(&bars.chroma[i], &bars.chroma[j]) + 0.5 * cosine(&bars.mel[i], &bars.mel[j])
}

fn build_neighbors(bars: &Bars, m: usize, floor: f32, top_k: usize) -> Vec<Vec<(usize, f32)>> {
    let mut neighbors: Vec<Vec<(usize, f32)>> = vec![Vec::new(); m];
    for (k, slot) in neighbors.iter_mut().enumerate() {
        let mut cand: Vec<(usize, f32)> = Vec::new();
        for j in 0..m {
            if j.abs_diff(k) < 2 {
                continue;
            }
            let s = bar_sim(bars, k, j);
            if s >= floor {
                cand.push((j, s));
            }
        }
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        cand.truncate(top_k);
        *slot = cand;
    }
    neighbors
}

/// DP over output bars: `cost[t*m + i]` is the cheapest way to be at source bar
/// `i` after `t` output bars.
#[allow(clippy::too_many_arguments)]
fn solve(
    neighbors: &[Vec<(usize, f32)>],
    m: usize,
    middle: usize,
    start: usize,
    outro_start: usize,
) -> (Vec<f32>, Vec<i32>) {
    let inf = f32::INFINITY;
    let mut cost = vec![inf; (middle + 1) * m];
    let mut prev = vec![-1i32; (middle + 1) * m];
    cost[start] = 0.0;
    for t in 1..=middle {
        for i in 0..m {
            let c = cost[(t - 1) * m + i];
            if !c.is_finite() {
                continue;
            }
            if i + 1 < m && i + 1 < outro_start {
                let idx = t * m + i + 1;
                if c < cost[idx] {
                    cost[idx] = c;
                    prev[idx] = i as i32;
                }
            }
            let k = i + 1;
            if k < m {
                for &(j, s) in &neighbors[k] {
                    if j >= outro_start {
                        continue;
                    }
                    let dist = (j as f32 - k as f32).abs() / m as f32;
                    let nc = c + (1.0 - s) + JUMP_PENALTY + DIST_PENALTY * dist * dist;
                    let idx = t * m + j;
                    if nc < cost[idx] {
                        cost[idx] = nc;
                        prev[idx] = i as i32;
                    }
                }
            }
        }
    }
    (cost, prev)
}

/// Find a remix path, or `None` if the material/structure is unsuitable.
#[allow(clippy::too_many_arguments)]
pub fn plan(
    chroma: &[f32],
    mel: &[f32],
    n_frames: usize,
    n_mels: usize,
    boundaries: &[usize],
    sample_rate: u32,
    hop: usize,
    trim_start: usize,
    trim_end: usize,
    target_samples: usize,
) -> Option<Vec<PlanSeg>> {
    let bars = build_bars(chroma, mel, n_frames, n_mels, boundaries);
    let m = bars.starts.len();
    if m < 8 {
        return None;
    }
    let target_secs = target_samples as f64 / sample_rate as f64;
    let bar_secs: Vec<f64> = (0..m)
        .map(|i| ((bars.ends[i] - bars.starts[i]) * hop) as f64 / sample_rate as f64)
        .collect();
    let mean_bar = (bar_secs.iter().sum::<f64>() / m as f64).max(0.2);

    let outro_bars = (m / 20).clamp(2, 16);
    let lead_bars = ((target_secs / mean_bar * 0.1).round() as usize).clamp(1, 16);
    if m < lead_bars + outro_bars + 3 {
        return None;
    }
    let outro_start = m - outro_bars;
    let start = lead_bars;
    let target_end = outro_start - 1;

    // Choose a neighbour set that can reach the song's end.
    let mid0 = ((target_secs / mean_bar).round() as usize)
        .saturating_sub(lead_bars + outro_bars)
        .max(1);
    let mut neighbors = build_neighbors(&bars, m, ATTEMPTS[0].0, ATTEMPTS[0].1);
    {
        let (cost, _) = solve(&neighbors, m, mid0, start, outro_start);
        if !cost[mid0 * m + target_end].is_finite() {
            for (floor, top_k) in ATTEMPTS.iter().skip(1) {
                let nb = build_neighbors(&bars, m, *floor, *top_k);
                let (cost, _) = solve(&nb, m, mid0, start, outro_start);
                neighbors = nb;
                if cost[mid0 * m + target_end].is_finite() {
                    break;
                }
            }
        }
    }

    let dbg = std::env::var_os("REMIX_DEBUG").is_some();

    // Converge on the target length by adjusting the number of output bars.
    let mut target_bars = ((target_secs / mean_bar).round() as usize).max(4);
    let mut best: Option<Vec<usize>> = None;
    for _ in 0..25 {
        let middle = target_bars.saturating_sub(lead_bars + outro_bars);
        if middle < 1 {
            break;
        }
        let (cost, prev) = solve(&neighbors, m, middle, start, outro_start);
        let mut end_i = target_end;
        if !cost[middle * m + target_end].is_finite() {
            let mut bc = f32::INFINITY;
            for i in 0..m {
                let c = cost[middle * m + i];
                if c.is_finite() && c < bc {
                    bc = c;
                    end_i = i;
                }
            }
            if !bc.is_finite() {
                break;
            }
        }

        // Backtrack.
        let mut path = vec![0usize; middle + 1];
        path[middle] = end_i;
        let mut i = end_i;
        let mut ok = true;
        for t in (1..=middle).rev() {
            let p = prev[t * m + i];
            if p < 0 {
                ok = false;
                break;
            }
            i = p as usize;
            path[t - 1] = i;
        }
        if !ok || path[0] != start {
            break;
        }

        let mut out_bars: Vec<usize> = (0..lead_bars).collect();
        out_bars.extend_from_slice(&path);
        out_bars.extend(outro_start..m);

        let total: f64 = out_bars.iter().map(|&b| bar_secs[b]).sum();
        if dbg {
            eprintln!(
                "[remix] target={target_secs:.2}s target_bars={target_bars} middle={middle} total={total:.2}s end={end_i}"
            );
        }
        best = Some(out_bars);
        if (total - target_secs).abs() <= 0.4 {
            break;
        }
        let delta = ((target_secs - total) / mean_bar).round() as i64;
        let new_bars = (target_bars as i64 + delta.clamp(-16, 16)).max(4) as usize;
        if new_bars == target_bars {
            break;
        }
        target_bars = new_bars;
    }
    let out_bars = best?;

    // Merge consecutive bars into segments; label runs.
    let samp = |f: usize| trim_start + f * hop;
    let mut segs: Vec<PlanSeg> = Vec::new();
    let mut idx = 0usize;
    let mut run = 0usize;
    while idx < out_bars.len() {
        let first = out_bars[idx];
        let mut end = idx + 1;
        while end < out_bars.len() && out_bars[end] == out_bars[end - 1] + 1 {
            end += 1;
        }
        let last = out_bars[end - 1];
        let is_first = run == 0;
        let is_last = end == out_bars.len();
        let kind = if is_first {
            "intro"
        } else if is_last {
            "outro"
        } else {
            "play"
        };
        let s_start = samp(bars.starts[first]);
        let s_end = if is_last { trim_end } else { samp(bars.ends[last]) };
        if s_end > s_start {
            segs.push(PlanSeg { kind, src_start: s_start, src_end: s_end });
        }
        idx = end;
        run += 1;
    }

    if segs.len() < 2 {
        return None;
    }
    Some(segs)
}
