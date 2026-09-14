use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum WavFormat {
    /// 32-bit IEEE float (default, lossless master)
    F32,
    /// 16-bit signed PCM
    S16,
    /// 24-bit signed PCM
    S24,
    /// 32-bit signed PCM
    S32,
}

#[derive(Parser, Debug, Clone)]
#[command(
    name = "remix",
    version,
    about = "Remix / loop a music track to a target length (single binary, no dependencies)"
)]
pub struct Cli {
    /// Input audio file (mp3, wav, flac, ogg, m4a/aac, ...)
    pub input: PathBuf,

    /// Target length: seconds ("95", "95.5") or "m:ss" or "h:mm:ss"
    #[arg(short, long)]
    pub length: String,

    /// Output WAV path (default: <input-stem>.remix.wav)
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// WAV sample format
    #[arg(long, value_enum, default_value_t = WavFormat::F32)]
    pub wav_format: WavFormat,

    /// Also write an MP3 here
    #[arg(long)]
    pub mp3: Option<PathBuf>,

    /// MP3 bitrate in kbps (CBR)
    #[arg(long, default_value_t = 320)]
    pub mp3_bitrate: u32,

    /// Also write an MP4 (waveform + markers + audio) here
    #[arg(long)]
    pub mp4: Option<PathBuf>,

    /// MP4 frame rate
    #[arg(long, default_value_t = 50)]
    pub fps: u32,

    /// MP4 size, e.g. 1920x1080
    #[arg(long, default_value = "1920x1080")]
    pub size: String,

    /// Do not append the detected outro (end by fading the last loop instead)
    #[arg(long)]
    pub no_outro: bool,

    /// Minimum loop length in seconds (default 4)
    #[arg(long)]
    pub min_loop: Option<f64>,

    /// Maximum loop length in seconds (default: the target length)
    #[arg(long)]
    pub max_loop: Option<f64>,

    /// Force loop start (seconds) instead of auto-detecting
    #[arg(long, requires = "loop_end")]
    pub loop_start: Option<f64>,

    /// Force loop end (seconds) instead of auto-detecting
    #[arg(long, requires = "loop_start")]
    pub loop_end: Option<f64>,

    /// Write a JSON timeline report here (default: <out-stem>.json when not quiet)
    #[arg(long)]
    pub report: Option<PathBuf>,

    /// Suppress progress output
    #[arg(short, long)]
    pub quiet: bool,
}

impl Cli {
    pub fn wav_path(&self) -> PathBuf {
        match &self.out {
            Some(p) => p.clone(),
            None => self.sibling("remix.wav"),
        }
    }

    pub fn report_path(&self) -> PathBuf {
        match &self.report {
            Some(p) => p.clone(),
            None => self.sibling("remix.json"),
        }
    }

    fn sibling(&self, suffix: &str) -> PathBuf {
        let stem = self
            .input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let parent = self.input.parent().unwrap_or_else(|| std::path::Path::new("."));
        parent.join(format!("{stem}.{suffix}"))
    }

    pub fn video_size(&self) -> anyhow::Result<(u32, u32)> {
        let (w, h) = self
            .size
            .split_once(['x', 'X'])
            .ok_or_else(|| anyhow::anyhow!("--size must be WxH, e.g. 1920x1080"))?;
        let w: u32 = w.trim().parse().map_err(|_| anyhow::anyhow!("bad width"))?;
        let h: u32 = h.trim().parse().map_err(|_| anyhow::anyhow!("bad height"))?;
        if w == 0 || h == 0 || !w.is_multiple_of(2) || !h.is_multiple_of(2) {
            anyhow::bail!("--size width/height must be non-zero even numbers (H.264)");
        }
        Ok((w, h))
    }
}

/// Parse "95", "95.5", "1:30", "0:01:30", "1:02:03.5" into seconds.
pub fn parse_duration(s: &str) -> anyhow::Result<f64> {
    let s = s.trim();
    if s.is_empty() {
        anyhow::bail!("empty length");
    }
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() > 3 {
        anyhow::bail!("length must be seconds, m:ss, or h:mm:ss");
    }
    let mut total = 0.0f64;
    for (i, p) in parts.iter().enumerate() {
        let v: f64 = p
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid length component: {p:?}"))?;
        let scale = match parts.len() - 1 - i {
            0 => 1.0,
            1 => 60.0,
            2 => 3600.0,
            _ => unreachable!(),
        };
        total += v * scale;
    }
    if !total.is_finite() || total <= 0.0 {
        anyhow::bail!("length must be a positive number of seconds");
    }
    Ok(total)
}
