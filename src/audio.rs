use anyhow::{Context, Result};
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decoded audio, de-interleaved per channel.
pub struct Audio {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f32>>,
}

impl Audio {
    pub fn frames(&self) -> usize {
        self.channels.first().map(|c| c.len()).unwrap_or(0)
    }

    pub fn n_channels(&self) -> usize {
        self.channels.len().max(1)
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }

    /// Channel-averaged mono signal.
    pub fn mono(&self) -> Vec<f32> {
        let n = self.frames();
        let nch = self.n_channels() as f32;
        let mut out = vec![0.0f32; n];
        for ch in &self.channels {
            for (o, s) in out.iter_mut().zip(ch.iter()) {
                *o += *s;
            }
        }
        let inv = 1.0 / nch;
        for v in out.iter_mut() {
            *v *= inv;
        }
        out
    }
}

pub fn decode(path: &Path) -> Result<Audio> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("cannot open input file: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .with_context(|| format!("unsupported or corrupt audio file: {}", path.display()))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("no decodable audio track found"))?;
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("no decoder available for this codec")?;

    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(44_100);
    let mut channels: Vec<Vec<f32>> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                sample_rate = spec.rate;
                if sample_buf.is_none() {
                    channels = vec![Vec::new(); spec.channels.count().max(1)];
                    sample_buf = Some(SampleBuffer::new(decoded.capacity() as u64, spec));
                }
                let sb = sample_buf.as_mut().unwrap();
                sb.copy_interleaved_ref(decoded);
                let nch = spec.channels.count().max(1);
                let samples = sb.samples();
                for (i, &v) in samples.iter().enumerate() {
                    let ch = i % nch;
                    if ch < channels.len() {
                        channels[ch].push(v);
                    }
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e).context("decode error"),
        }
    }

    if channels.is_empty() {
        anyhow::bail!("no audio samples decoded");
    }
    // Normalise channel lengths (defensive).
    let n = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    for c in channels.iter_mut() {
        c.truncate(n);
    }

    Ok(Audio {
        sample_rate,
        channels,
    })
}
