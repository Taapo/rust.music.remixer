mod assemble;
mod audio;
mod cli;
mod dsp;
mod encode;
mod loops;
mod render;
mod report;
mod video;

use anyhow::{Context, Result};
use clap::Parser;

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
    let (bpm, raw_beats) = dsp::detect_beats(&cli.input, sr, dsp::HOP, full_frames);
    let mut beats: Vec<usize> = raw_beats
        .iter()
        .filter_map(|&b| b.checked_sub(trim_start_frames))
        .filter(|&b| b < n_frames)
        .collect();
    beats.sort_unstable();
    beats.dedup();
    let used_grid = if beats.len() < 4 {
        beats = dsp::synth_beats(bpm, sr, dsp::HOP, n_frames);
        true
    } else {
        false
    };
    if verbose {
        println!(
            "analysis: trimmed {:.2}s..{:.2}s, {} beats @ {:.1} bpm{}",
            trim_start as f64 / sr as f64,
            trim_end as f64 / sr as f64,
            beats.len(),
            bpm,
            if used_grid { " (synthetic grid)" } else { "" }
        );
    }

    // --- Loop finding ---
    let min_loop_frames = ((0.35 * n_frames as f32) as usize).max(1);
    let pairs = loops::find_best_loop_points(
        &features.chroma,
        &features.power_db,
        features.n_bins,
        n_frames,
        &beats,
        bpm,
        sr,
        min_loop_frames,
        n_frames,
        false,
    );

    let (loop_start_frame, loop_end_frame) = if let Some(best) = pairs.first().cloned() {
        (best.start_frame, best.end_frame)
    } else {
        if verbose {
            eprintln!("warning: no loop point found; looping the whole trimmed track");
        }
        (0usize, n_frames.saturating_sub(1).max(1))
    };

    // Trimmed frames -> original samples, then snap to zero crossings.
    let loop_start_orig = loop_start_frame * dsp::HOP + trim_start;
    let loop_end_orig = loop_end_frame * dsp::HOP + trim_start;
    let loop_start = loops::nearest_zero_crossing(&mono, sr, loop_start_orig.min(audio.frames() - 1));
    let loop_end = loops::nearest_zero_crossing(&mono, sr, loop_end_orig.min(audio.frames() - 1));

    if verbose {
        println!(
            "loop   : {:.3}s .. {:.3}s  (len {:.3}s){}",
            loop_start as f64 / sr as f64,
            loop_end as f64 / sr as f64,
            (loop_end - loop_start) as f64 / sr as f64,
            pairs
                .first()
                .map(|p| format!("  score {:.4}", p.score))
                .unwrap_or_default()
        );
    }

    // --- Assembly ---
    let target_samples = (target_secs * sr as f64).round() as usize;
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
