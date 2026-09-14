//! Builds the remixed output by assembling intro + loop×N + outro to a target
//! length, with short equal-linear crossfades at every seam.

use crate::audio::Audio;

#[derive(Clone, Debug)]
pub struct Segment {
    /// "intro" | "loop" | "outro" | "tail"
    pub kind: &'static str,
    pub out_start: usize,
    pub out_end: usize,
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

/// `loop_start`/`loop_end`/`trim_end` are sample indices into the original file.
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
    let loop_len = loop_end.saturating_sub(loop_start);
    let outro_len = if include_outro {
        trim_end.saturating_sub(loop_end)
    } else {
        0
    };

    // Build the segment plan: (kind, src_start, src_end).
    // Layout: intro + N full loops + optional partial-loop filler + outro.
    let mut plan: Vec<(&'static str, usize, usize)> = Vec::new();
    if loop_start > 0 {
        plan.push(("intro", 0, loop_start));
    }

    let min_filler = (0.25 * audio.sample_rate as f32) as usize;
    let end_outro = if include_outro { outro_len } else { 0 };

    if loop_len > 0 && target_samples > loop_start + end_outro {
        let space = target_samples - loop_start - end_outro;
        let n = space / loop_len;
        let filler = space % loop_len;
        for _ in 0..n {
            plan.push(("loop", loop_start, loop_end));
        }
        if filler >= min_filler {
            plan.push(("tail", loop_start, (loop_start + filler).min(loop_end)));
        }
        if include_outro && outro_len > 0 {
            plan.push(("outro", loop_end, trim_end));
        }
    } else {
        let end = target_samples.min(trim_end).max(1);
        plan.push(("tail", 0, end));
    }

    // Render each channel with the same plan.
    let nch = audio.n_channels();
    let mut channels: Vec<Vec<f32>> = vec![Vec::new(); nch];
    let mut segments: Vec<Segment> = Vec::new();

    for (kind, s_start, s_end) in &plan {
        let before = channels[0].len();
        for (c, out) in channels.iter_mut().enumerate() {
            let src_ch = audio
                .channels
                .get(c)
                .or_else(|| audio.channels.first())
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let a = (*s_start).min(src_ch.len());
            let b = (*s_end).min(src_ch.len());
            let slice = &src_ch[a..b.max(a)];
            crossfade_append(out, slice, xfade);
        }
        let after = channels[0].len();
        segments.push(Segment {
            kind,
            out_start: before,
            out_end: after,
            src_start: *s_start,
            src_end: *s_end,
        });
    }

    // Fade the very end only when the track ends on a truncated loop.
    let fade = (0.5 * audio.sample_rate as f32) as usize;
    let last_kind = segments.last().map(|s| s.kind).unwrap_or("tail");
    if last_kind == "tail" {
        for ch in channels.iter_mut() {
            apply_fade_out(ch, fade);
        }
    }

    // Land on the target length.
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
