mod assemble;
mod audio;
mod beats;
mod cli;
mod dsp;
mod encode;
mod loops;
mod render;
mod report;
mod video;

use anyhow::{Context, Result};
use clap::Parser;

fn cand_line(p: &loops::LoopPair, trim_start: usize, sr: u32) -> String {
    format!(
        "    {:.2}s..{:.2}s (len {:.2}s)  note {:.3}  loud {:.3}  score {:.4}",
        (p.start_frame * dsp::HOP + trim_start) as f64 / sr as f64,
        (p.end_frame * dsp::HOP + trim_start) as f64 / sr as f64,
        ((p.end_frame - p.start_frame) * dsp::HOP) as f64 / sr as f64,
        p.note_distance,
        p.loudness_difference,
        p.score
    )
}

fn main() -> Result<()> {
    let cli = cli::Cli::parse();
    let target_secs = cli::parse_duration(&cli.length)?;
    let verbose = !cli.quiet;

    let audio = audio::decode(&cli.input)?;
    if audio.frames() == 0 {
        anyhow::bail!("input contains no audio samples");
    }
    let sr = audio.sample_rate;

    if verbose {
        println!(
            "input : {}  ({:.2}s, {} Hz, {} ch)",
            cli.input.display(),
            audio.duration_secs(),
            sr,
            audio.n_channels()
        );
    }

    // --- Analysis (on the trimmed, normalised mono signal) ---
    let mut mono = audio.mono();
    let max_abs = mono.iter().cloned().fold(0.0f32, f32::max).abs();
    if max_abs > 0.0 {
        let inv = 1.0 / max_abs;
        for v in mono.iter_mut() {
            *v *= inv;
        }
    }

    let (trim_start, trim_end) = dsp::trim_silence(&mono, 40.0);
    let trimmed = &mono[trim_start..trim_end];
    if trimmed.len() < dsp::N_FFT {
        anyhow::bail!("track is too short to analyse after trimming silence");
    }
    let features = dsp::compute_features(trimmed, sr, dsp::N_FFT, dsp::HOP);
    let n_frames = features.n_frames;
    let trim_start_frames = trim_start / dsp::HOP;

    let full_frames = 1 + mono.len() / dsp::HOP;
    let map_times = |times: &[f32]| -> Vec<usize> {
        let mut v: Vec<usize> = times
            .iter()
            .map(|&t| (t * sr as f32 / dsp::HOP as f32).round() as i64)
            .filter_map(|i| {
                let j = i - trim_start_frames as i64;
                (j >= 0 && (j as usize) < n_frames).then_some(j as usize)
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };

    let (bpm, beat_frames, downbeat_frames) = match beats::analyze(&cli.input) {
        Ok(g) => (
            g.bpm,
            map_times(&g.beats_secs),
            map_times(&g.downbeats_secs),
        ),
        Err(e) => {
            if verbose {
                eprintln!("note   : Beat This! unavailable ({e}); using fallback tracker");
            }
            let (bpm, raw) = dsp::detect_beats(&cli.input, sr, dsp::HOP, full_frames);
            let frames = map_times(
                &raw.iter()
                    .map(|&f| f as f32 * dsp::HOP as f32 / sr as f32)
                    .collect::<Vec<_>>(),
            );
            (bpm, frames, Vec::new())
        }
    };

    // Candidates come from beats (more options); downbeats are used to prefer
    // musically aligned (bar-start) loops.
    let mut beats = beat_frames.clone();
    let boundary_kind = "beats";
    let used_grid = if beats.len() < 4 {
        beats = dsp::synth_beats(bpm, sr, dsp::HOP, n_frames);
        true
    } else {
        false
    };
    if verbose {
        println!(
            "analysis: trimmed {:.2}s..{:.2}s, {:.1} bpm, {} {}, {} downbeats{}",
            trim_start as f64 / sr as f64,
            trim_end as f64 / sr as f64,
            bpm,
            beats.len(),
            boundary_kind,
            downbeat_frames.len(),
            if used_grid { " (synthetic grid)" } else { "" }
        );
    }

    // --- Loop finding ---
    let trimmed_secs = trimmed.len() as f64 / sr as f64;
    let target_samples = (target_secs * sr as f64).round() as usize;
    let min_loop_seconds = cli.min_loop.unwrap_or(4.0).max(0.5);
    let max_loop_seconds = cli
        .max_loop
        .unwrap_or(target_secs)
        .min(trimmed_secs)
        .max(min_loop_seconds);
    let min_loop_frames = ((min_loop_seconds * sr as f64 / dsp::HOP as f64) as usize).max(1);
    let max_loop_frames = ((max_loop_seconds * sr as f64 / dsp::HOP as f64) as usize)
        .min(n_frames)
        .max(min_loop_frames + 1);
    if verbose {
        println!(
            "loops  : searching {:.1}s..{:.1}s (track {:.1}s, target {:.1}s)",
            min_loop_seconds, max_loop_seconds, trimmed_secs, target_secs
        );
    }
    let (loop_start, loop_end, chosen_score) =
        if let (Some(ls), Some(le)) = (cli.loop_start, cli.loop_end) {
            let a = ((ls.max(0.0) * sr as f64).round() as usize).min(audio.frames() - 1);
            let b = (le.max(0.0) * sr as f64).round() as usize;
            let b = b.min(audio.frames());
            anyhow::ensure!(
                b > a + sr as usize / 20,
                "--loop-end must be at least 50 ms after --loop-start"
            );
            (
                loops::nearest_zero_crossing(&mono, sr, a),
                loops::nearest_zero_crossing(&mono, sr, b),
                1.0f32,
            )
        } else {
            let pairs = loops::find_best_loop_points(
                &features.chroma,
                &features.mel,
                dsp::N_MELS,
                &features.power_db,
                features.n_bins,
                n_frames,
                &beats,
                bpm,
                sr,
                min_loop_frames,
                max_loop_frames,
                false,
            );
            // Best loop short enough to repeat, with a small bonus for loops
            // that begin/end on a bar start (downbeat).
            let db = &downbeat_frames;
            let near_db = |f: usize| db.iter().any(|&d| d.abs_diff(f) <= 8);
            let rank = |p: &loops::LoopPair| {
                let mut s = p.score;
                if near_db(p.start_frame) {
                    s += 0.01;
                }
                if near_db(p.end_frame) {
                    s += 0.01;
                }
                s
            };
            let eligible: Vec<&loops::LoopPair> = pairs
                .iter()
                .filter(|p| (p.end_frame - p.start_frame) * dsp::HOP <= target_samples)
                .collect();
            let chosen = eligible
                .iter()
                .copied()
                .max_by(|a, b| rank(a).partial_cmp(&rank(b)).unwrap())
                .or_else(|| pairs.first());
            let (sf, ef, score) = match chosen {
                Some(best) => (best.start_frame, best.end_frame, best.score),
                None => {
                    if verbose {
                        eprintln!("warning: no loop point found; looping the whole trimmed track");
                    }
                    (0usize, n_frames.saturating_sub(1).max(1), 0.0)
                }
            };
            if verbose {
                println!("  top candidates:");
                for p in pairs.iter().take(8) {
                    println!("{}", cand_line(p, trim_start, sr));
                }
            }
            let so = sf * dsp::HOP + trim_start;
            let eo = ef * dsp::HOP + trim_start;
            (
                loops::nearest_zero_crossing(&mono, sr, so.min(audio.frames() - 1)),
                loops::nearest_zero_crossing(&mono, sr, eo.min(audio.frames() - 1)),
                score,
            )
        };

    if verbose {
        println!(
            "loop   : {:.3}s .. {:.3}s  (len {:.3}s){}",
            loop_start as f64 / sr as f64,
            loop_end as f64 / sr as f64,
            (loop_end - loop_start) as f64 / sr as f64,
            if cli.loop_start.is_some() {
                "  (manual)".to_string()
            } else {
                format!("  score {chosen_score:.4}")
            },
        );
    }

    // --- Assembly ---
    let xfade = (0.012 * sr as f64) as usize;
    let assembly = assemble::assemble(
        &audio,
        loop_start,
        loop_end,
        trim_end,
        target_samples,
        !cli.no_outro,
        xfade,
    );
    if verbose {
        println!(
            "output : {:.2}s (target {:.2}s), {} segments",
            assembly.duration_secs(),
            target_secs,
            assembly.segments.len()
        );
    }

    // --- Report ---
    let rep = report::build(
        &assembly,
        &cli.input.display().to_string(),
        target_secs,
        bpm,
        loop_start,
        loop_end,
        trim_start,
        trim_end,
    );
    if verbose {
        println!("\ntimeline ({} segments):", rep.segments.len());
        for s in &rep.segments {
            println!(
                "  {:<6} out {:.2}s..{:.2}s  <- source {:.2}s..{:.2}s",
                s.kind, s.out_start, s.out_end, s.source_start, s.source_end
            );
        }
    }

    // --- Outputs ---
    let wav_path = cli.wav_path();
    encode::write_wav(&wav_path, &assembly, cli.wav_format)?;
    if verbose {
        println!("wrote  : {}", wav_path.display());
    }
    if let Some(mp3) = &cli.mp3 {
        encode::write_mp3(mp3, &assembly, cli.mp3_bitrate)?;
        if verbose {
            println!("wrote  : {}", mp3.display());
        }
    }
    if let Some(mp4) = &cli.mp4 {
        let (w, h) = cli.video_size()?;
        video::write_mp4(mp4, &assembly, &rep, cli.fps, w, h)
            .with_context(|| format!("failed to write MP4: {}", mp4.display()))?;
        if verbose {
            println!("wrote  : {}", mp4.display());
        }
    }

    let report_path = cli.report_path();
    rep.write_json(&report_path)?;
    if verbose {
        println!("report : {}", report_path.display());
    }

    Ok(())
}
