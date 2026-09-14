use crate::assemble::Assembly;
use serde::Serialize;

#[derive(Serialize)]
pub struct ReportSegment {
    pub kind: String,
    pub out_start: f64,
    pub out_end: f64,
    pub source_start: f64,
    pub source_end: f64,
}

#[derive(Serialize)]
pub struct Report {
    pub input: String,
    pub target_secs: f64,
    pub result_secs: f64,
    pub sample_rate: u32,
    pub bpm: f32,
    pub loop_start_secs: f64,
    pub loop_end_secs: f64,
    pub trimmed_start_secs: f64,
    pub trimmed_end_secs: f64,
    pub loop_iterations: usize,
    pub segments: Vec<ReportSegment>,
}

#[allow(clippy::too_many_arguments)]
pub fn build(
    assembly: &Assembly,
    input: &str,
    target_secs: f64,
    bpm: f32,
    loop_start: usize,
    loop_end: usize,
    trim_start: usize,
    trim_end: usize,
) -> Report {
    let sr = assembly.sample_rate as f64;
    let segments = assembly
        .segments
        .iter()
        .map(|s| ReportSegment {
            kind: s.kind.to_string(),
            out_start: s.out_start as f64 / sr,
            out_end: s.out_end as f64 / sr,
            source_start: s.src_start as f64 / sr,
            source_end: s.src_end as f64 / sr,
        })
        .collect();
    let loop_iterations = assembly
        .segments
        .iter()
        .filter(|s| s.kind == "loop")
        .count();
    Report {
        input: input.to_string(),
        target_secs,
        result_secs: assembly.duration_secs(),
        sample_rate: assembly.sample_rate,
        bpm,
        loop_start_secs: loop_start as f64 / sr,
        loop_end_secs: loop_end as f64 / sr,
        trimmed_start_secs: trim_start as f64 / sr,
        trimmed_end_secs: trim_end as f64 / sr,
        loop_iterations,
        segments,
    }
}

impl Report {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn write_json(&self, path: &std::path::Path) -> anyhow::Result<()> {
        std::fs::write(path, self.to_json())
            .map_err(|e| anyhow::anyhow!("cannot write report {}: {e}", path.display()))?;
        Ok(())
    }
}
