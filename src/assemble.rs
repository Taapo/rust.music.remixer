//! Builds the remixed output by rendering an ordered list of source segments
//! with short equal-linear crossfades at every seam.

use crate::audio::Audio;

#[derive(Clone, Debug)]
pub struct Segment {
    /// e.g. "intro" | "play" | "jump" | "outro" | "loop" | "tail"
    pub kind: &'static str,
    pub out_start: usize,
    pub out_end: usize,
    pub src_start: usize,
    pub src_end: usize,
}

/// One entry of a plan: a source range (original-file samples) and a label.
#[derive(Clone, Debug)]
pub struct PlanSeg {
    pub kind: &'static str,
    pub src_start: usize,
    pub src_end: usize,
}

pub struct Assembly {
    pub channels: Vec<Vec<f32>>,
    pub segments: Vec<Segment>,
    pub sample_rate: u32,
}

impl Assembly {
    pub fn frames(&self) -> usize {
        self.channels.first().map(|c| c.len()).unwrap_or(0)
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }
}

fn crossfade_append(dst: &mut Vec<f32>, src: &[f32], xfade: usize) {
    if src.is_empty() {
        return;
    }
    if dst.is_empty() {
        dst.extend_from_slice(src);
        return;
    }
    let x = xfade.min(dst.len()).min(src.len());
    if x == 0 {
        dst.extend_from_slice(src);
        return;
    }
    let base = dst.len() - x;
    for i in 0..x {
        let t = (i + 1) as f32 / (x + 1) as f32;
        dst[base + i] = dst[base + i] * (1.0 - t) + src[i] * t;
    }
    dst.extend_from_slice(&src[x..]);
}

fn apply_fade_out(channel: &mut [f32], fade: usize) {
    let n = channel.len();
    let f = fade.min(n);
    if f == 0 {
        return;
    }
    let start = n - f;
    for i in 0..f {
        let gain = 1.0 - (i as f32 / f as f32);
        channel[start + i] *= gain;
    }
}

/// Render a plan of source segments to the target length.
pub fn render(
    audio: &Audio,
    plan: &[PlanSeg],
    target_samples: usize,
    xfade: usize,
    fade_end: bool,
) -> Assembly {
    let nch = audio.n_channels();
    let mut channels: Vec<Vec<f32>> = vec![Vec::new(); nch];
    let mut segments: Vec<Segment> = Vec::new();

    for seg in plan {
        let before = channels[0].len();
        for (c, out) in channels.iter_mut().enumerate() {
            let src_ch = audio
                .channels
                .get(c)
                .or_else(|| audio.channels.first())
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let a = seg.src_start.min(src_ch.len());
            let b = seg.src_end.min(src_ch.len());
            crossfade_append(out, &src_ch[a..b.max(a)], xfade);
        }
        segments.push(Segment {
            kind: seg.kind,
            out_start: before,
            out_end: channels[0].len(),
            src_start: seg.src_start,
            src_end: seg.src_end,
        });
    }

    let fade = (0.6 * audio.sample_rate as f32) as usize;
    if fade_end {
        for ch in channels.iter_mut() {
            apply_fade_out(ch, fade);
        }
    }

    if channels[0].len() > target_samples {
        for ch in channels.iter_mut() {
            ch.truncate(target_samples);
        }
        if let Some(last) = segments.last_mut() {
            last.out_end = last.out_end.min(target_samples);
        }
    }

    Assembly {
        channels,
        segments,
        sample_rate: audio.sample_rate,
    }
}

/// Classic single-loop remix: `lead-in -> loop×N -> (outro | partial loop + fade)`.
/// `loop_start`/`loop_end`/`trim_end` are sample indices into the original file.
/// Used for the manual `--loop-start/--loop-end` override and as a fallback.
#[allow(clippy::too_many_arguments)]
pub fn assemble(
    audio: &Audio,
    loop_start: usize,
    loop_end: usize,
    trim_end: usize,
    target_samples: usize,
    include_outro: bool,
    xfade: usize,
) -> Assembly {
    let loop_end = loop_end.min(audio.frames());
    let trim_end = trim_end.min(audio.frames());
    let intro_len = loop_start.min(audio.frames());
    let loop_len = loop_end.saturating_sub(intro_len);
    let outro_len = if include_outro {
        trim_end.saturating_sub(loop_end)
    } else {
        0
    };

    let mut plan: Vec<PlanSeg> = Vec::new();

    if loop_len == 0 || target_samples < loop_len {
        let end = target_samples.min(trim_end).max(1);
        plan.push(PlanSeg { kind: "tail", src_start: 0, src_end: end });
        return render(audio, &plan, target_samples, xfade, true);
    }

    let lead_in = if intro_len + loop_len <= target_samples {
        intro_len
    } else {
        let cap = target_samples / 5 * 2;
        intro_len.min(target_samples - loop_len).min(cap)
    };
    let intro_start = intro_len - lead_in;
    if lead_in > 0 {
        plan.push(PlanSeg { kind: "intro", src_start: intro_start, src_end: intro_len });
    }
    let remaining = target_samples - lead_in;
    if include_outro && outro_len > 0 && remaining >= loop_len + outro_len {
        let k = (remaining - outro_len) / loop_len;
        for _ in 0..k {
            plan.push(PlanSeg { kind: "loop", src_start: intro_len, src_end: loop_end });
        }
        plan.push(PlanSeg { kind: "outro", src_start: loop_end, src_end: trim_end });
        return render(audio, &plan, target_samples, xfade, false);
    }
    let k = remaining / loop_len;
    let rem = remaining % loop_len;
    for _ in 0..k {
        plan.push(PlanSeg { kind: "loop", src_start: intro_len, src_end: loop_end });
    }
    if rem > 0 {
        let end = (intro_len + rem).min(loop_end);
        if end > intro_len {
            plan.push(PlanSeg { kind: "tail", src_start: intro_len, src_end: end });
        }
    }
    render(audio, &plan, target_samples, xfade, true)
}
