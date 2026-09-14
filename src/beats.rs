//! Beat and downbeat detection using the "Beat This!" model (ISMIR 2024) via
//! the pure-Rust `rten` runtime. The two ONNX models are embedded in the
//! binary, so there are no external files to ship; they are materialised to a
//! temp file on first use because the runtime loads models from paths.

use anyhow::{Context, Result};
use beat_this::{calculate_bpm, BeatThis, RtenRuntime};
use std::path::{Path, PathBuf};

static MEL_MODEL: &[u8] = include_bytes!("../models/mel_spectrogram.onnx");
static BEAT_MODEL: &[u8] = include_bytes!("../models/beat_this_small.onnx");

pub struct BeatGrid {
    pub beats_secs: Vec<f32>,
    pub downbeats_secs: Vec<f32>,
    pub bpm: f32,
}

fn materialize(name: &str, bytes: &[u8]) -> Result<PathBuf> {
    let mut p = std::env::temp_dir();
    p.push(name);
    let stale = match std::fs::metadata(&p) {
        Ok(m) => m.len() != bytes.len() as u64,
        Err(_) => true,
    };
    if stale {
        std::fs::write(&p, bytes)
            .with_context(|| format!("cannot write embedded model to {}", p.display()))?;
    }
    Ok(p)
}

pub fn analyze(path: &Path) -> Result<BeatGrid> {
    let mel = materialize("remix_mel_spectrogram.onnx", MEL_MODEL)?;
    let beat = materialize("remix_beat_this_small.onnx", BEAT_MODEL)?;

    let runtime = RtenRuntime;
    let mut tracker =
        BeatThis::new(&runtime, &mel, &beat).context("failed to load Beat This! models")?;
    let analysis = tracker
        .analyze_file(path)
        .context("Beat This! analysis failed")?;
    let bpm = calculate_bpm(&analysis).unwrap_or(120.0);

    Ok(BeatGrid {
        beats_secs: analysis.beats.clone(),
        downbeats_secs: analysis.downbeats.clone(),
        bpm,
    })
}
