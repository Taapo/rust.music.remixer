//! MP4 output: H.264 video (OpenH264, statically linked) muxed with AAC-LC
//! audio (pure-Rust encoder) via the pure-Rust `muxide` muxer.

use crate::assemble::Assembly;
use crate::render;
use crate::report::Report;
use anyhow::{Context, Result};
use muxide::api::{AacProfile, AudioCodec, MuxerBuilder, VideoCodec};
use openh264::encoder::{Encoder, FrameType};
use openh264::formats::{RgbaSliceU8, YUVBuffer};
use rusty_aac::{write_adts_header, AacEncoder, AacEncoderConfig, AdtsHeader};
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

pub fn write_mp4(
    path: &Path,
    assembly: &Assembly,
    report: &Report,
    fps: u32,
    w: u32,
    h: u32,
) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("cannot create MP4: {}", path.display()))?;
    let writer = BufWriter::new(file);

    let (aac_channels, interleaved, sample_rate) = aac_pcm(assembly);
    let mut muxer = MuxerBuilder::new(writer)
        .video(VideoCodec::H264, w, h, fps as f64)
        .audio(
            AudioCodec::Aac(AacProfile::Lc),
            sample_rate,
            aac_channels as u16,
        )
        .with_fast_start(true)
        .build()
        .context("failed to start MP4 muxer")?;

    // Render the static background once; each frame clones it and stamps a playhead.
    let bg = render::background(assembly, report, w as usize, h as usize);
    let total_secs = assembly.duration_secs().max(1.0 / fps as f64);
    let total_frames = (total_secs * fps as f64).ceil().max(1.0) as usize;

    let mut encoder = Encoder::new().context("failed to create H.264 encoder")?;
    for f in 0..total_frames {
        let t = f as f64 / fps as f64;
        let mut frame = bg.clone();
        render::draw_playhead(&mut frame, t / total_secs);

        let yuv = YUVBuffer::from_rgba8_source(RgbaSliceU8::new(
            &frame.px,
            (w as usize, h as usize),
        ));
        let bitstream = encoder
            .encode(&yuv)
            .map_err(|e| anyhow::anyhow!("H.264 encode error: {e}"))?;
        let key = matches!(bitstream.frame_type(), FrameType::IDR | FrameType::I);
        let mut data = Vec::new();
        bitstream.write_vec(&mut data);
        if !data.is_empty() {
            muxer
                .write_video(t, &data, key)
                .map_err(|e| anyhow::anyhow!("mux video: {e}"))?;
        }
    }

    // Audio after the first video frame (muxide requirement).
    encode_audio(&mut muxer, &interleaved, aac_channels as u16, sample_rate, 320_000)?;

    muxer
        .finish()
        .map_err(|e| anyhow::anyhow!("finish MP4: {e}"))?;
    Ok(())
}

/// Pick AAC channels (mono/stereo) and build interleaved f32 PCM.
fn aac_pcm(assembly: &Assembly) -> (usize, Vec<f32>, u32) {
    let nch = assembly.channels.len().max(1);
    let n = assembly.frames();
    if nch == 1 {
        (1, assembly.channels[0].clone(), assembly.sample_rate)
    } else {
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            out.push(assembly.channels[0][i]);
            out.push(assembly.channels[1][i]);
        }
        (2, out, assembly.sample_rate)
    }
}

fn encode_audio<W: Write + Seek>(
    muxer: &mut muxide::api::Muxer<W>,
    interleaved: &[f32],
    channels: u16,
    sample_rate: u32,
    bitrate_bps: u32,
) -> Result<()> {
    let mut enc = AacEncoder::new(AacEncoderConfig {
        bitrate_bps,
        ..Default::default()
    });
    enc.push_pcm(interleaved, channels, sample_rate)
        .map_err(|e| anyhow::anyhow!("AAC encode: {e:?}"))?;
    enc.finish();

    while let Ok(p) = enc.next_packet() {
        let hdr = AdtsHeader {
            object_type: 2,
            sample_rate,
            channels,
            frame_length: 7 + p.data.len(),
            header_len: 7,
        };
        let mut frame = write_adts_header(&hdr).to_vec();
        frame.extend_from_slice(&p.data);
        let pts = p.pts as f64 / sample_rate as f64;
        muxer
            .write_audio(pts, &frame)
            .map_err(|e| anyhow::anyhow!("mux audio: {e}"))?;
    }
    Ok(())
}
