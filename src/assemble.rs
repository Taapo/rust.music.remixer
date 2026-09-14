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
    let intro_len = loop_start.min(audio.frames());
    let loop_len = loop_end.saturating_sub(intro_len);
    let outro_len = if include_outro {
        trim_end.saturating_sub(loop_end)
    } else {
        0
    };

    // Build the segment plan: (kind, src_start, src_end).
    // Only seamless structures are used: lead-in -> [loop]* -> (outro | fade).
    // The outro may only follow the end of a full loop, never a fragment.
    let mut plan: Vec<(&'static str, usize, usize)> = Vec::new();

    if loop_len == 0 || target_samples < loop_len {
        // No usable loop (or target too short for even one loop): cut + fade.
        let end = target_samples.min(trim_end).max(1);
        plan.push(("tail", 0, end));
    } else {
        // Lead-in: if the whole intro + one loop fits, start at the beginning.
        // Otherwise start later so the loop still gets the bulk of the runtime.
        let lead_in = if intro_len + loop_len <= target_samples {
            intro_len
        } else {
            let cap = target_samples / 5 * 2; // ~40% lead-in
            intro_len.min(target_samples - loop_len).min(cap)
        };
        let intro_start = intro_len - lead_in;
        if lead_in > 0 {
            plan.push(("intro", intro_start, intro_len));
        }
        let remaining = target_samples - lead_in;
        if include_outro && outro_len > 0 && remaining >= loop_len + outro_len {
            // lead-in + k full loops + outro (natural ending).
            let k = (remaining - outro_len) / loop_len;
            for _ in 0..k {
                plan.push(("loop", intro_len, loop_end));
            }
            plan.push(("outro", loop_end, trim_end));
        } else {
            // lead-in + full loops + a faded partial loop to land on target.
            let k = remaining / loop_len;
            let rem = remaining % loop_len;
            for _ in 0..k {
                plan.push(("loop", intro_len, loop_end));
            }
            if rem > 0 {
                let end = (intro_len + rem).min(loop_end);
                if end > intro_len {
                    plan.push(("tail", intro_len, end));
                }
            }
        }
    }

    let ends_on_outro = plan.last().map(|s| s.0 == "outro").unwrap_or(false);

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

    // Fade the end unless it lands on the natural outro.
    let fade = (0.6 * audio.sample_rate as f32) as usize;
    if !ends_on_outro {
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
