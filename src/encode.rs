use crate::assemble::Assembly;
use crate::cli::WavFormat;
use anyhow::{Context, Result};
use mp3lame_encoder::{Bitrate, Builder, FlushNoGap, MonoPcm};
use std::path::Path;

pub fn write_wav(path: &Path, a: &Assembly, fmt: WavFormat) -> Result<()> {
    let channels = a.channels.len().max(1) as u16;
    let (bits, sample_format) = match fmt {
        WavFormat::F32 => (32, hound::SampleFormat::Float),
        WavFormat::S16 => (16, hound::SampleFormat::Int),
        WavFormat::S24 => (24, hound::SampleFormat::Int),
        WavFormat::S32 => (32, hound::SampleFormat::Int),
    };
    let spec = hound::WavSpec {
        channels,
        sample_rate: a.sample_rate,
        bits_per_sample: bits,
        sample_format,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("cannot create WAV: {}", path.display()))?;
    let frames = a.frames();
    let nch = a.channels.len().max(1);
    for i in 0..frames {
        for ch in 0..nch {
            let v = a.channels[ch][i];
            match fmt {
                WavFormat::F32 => writer.write_sample(v)?,
                WavFormat::S16 => {
                    writer.write_sample((v.clamp(-1.0, 1.0) * 32767.0) as i16)?
                }
                WavFormat::S24 => {
                    writer.write_sample((v.clamp(-1.0, 1.0) * 8_388_607.0) as i32)?
                }
                WavFormat::S32 => {
                    writer.write_sample((v.clamp(-1.0, 1.0) * 2_147_483_647.0) as i32)?
                }
            }
        }
    }
    writer.finalize().context("failed to finalise WAV")?;
    Ok(())
}

fn bitrate_from_kbps(kbps: u32) -> Bitrate {
    match kbps {
        8 => Bitrate::Kbps8,
        16 => Bitrate::Kbps16,
        24 => Bitrate::Kbps24,
        32 => Bitrate::Kbps32,
        40 => Bitrate::Kbps40,
        48 => Bitrate::Kbps48,
        64 => Bitrate::Kbps64,
        80 => Bitrate::Kbps80,
        96 => Bitrate::Kbps96,
        112 => Bitrate::Kbps112,
        128 => Bitrate::Kbps128,
        160 => Bitrate::Kbps160,
        192 => Bitrate::Kbps192,
        224 => Bitrate::Kbps224,
        256 => Bitrate::Kbps256,
        _ => Bitrate::Kbps320,
    }
}

pub fn write_mp3(path: &Path, a: &Assembly, bitrate_kbps: u32) -> Result<()> {
    let nch = a.channels.len().max(1);
    let mut builder = Builder::new().context("failed to create LAME encoder")?;
    builder
        .set_num_channels(nch as u8)
        .context("LAME: set channels")?;
    builder
        .set_sample_rate(a.sample_rate)
        .context("LAME: set sample rate")?;
    builder
        .set_brate(bitrate_from_kbps(bitrate_kbps))
        .context("LAME: set bitrate")?;
    builder
        .set_quality(mp3lame_encoder::Quality::Best)
        .context("LAME: set quality")?;
    let mut encoder = builder.build().context("LAME: build")?;

    // Interleave float samples.
    let frames = a.frames();
    let mut interleaved = Vec::with_capacity(frames * nch);
    for i in 0..frames {
        for ch in 0..nch {
            interleaved.push(a.channels[ch][i]);
        }
    }

    let mut out = Vec::new();
    for chunk in interleaved.chunks(1152 * nch) {
        out.reserve(mp3lame_encoder::max_required_buffer_size(chunk.len()));
        if nch == 1 {
            encoder
                .encode_to_vec(MonoPcm(chunk), &mut out)
                .map_err(|e| anyhow::anyhow!("LAME encode: {e:?}"))?;
        } else {
            encoder
                .encode_to_vec(mp3lame_encoder::InterleavedPcm(chunk), &mut out)
                .map_err(|e| anyhow::anyhow!("LAME encode: {e:?}"))?;
        }
    }
    out.reserve(mp3lame_encoder::max_required_buffer_size(1152 * nch));
    encoder
        .flush_to_vec::<FlushNoGap>(&mut out)
        .map_err(|e| anyhow::anyhow!("LAME flush: {e:?}"))?;

    std::fs::write(path, &out)
        .with_context(|| format!("cannot write MP3: {}", path.display()))?;
    Ok(())
}
