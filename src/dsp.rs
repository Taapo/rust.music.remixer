use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::path::Path;

pub const N_FFT: usize = 2048;
pub const HOP: usize = 512;
pub const N_MELS: usize = 24;

/// Trim leading/trailing silence. Returns (start_sample, end_sample) in `mono`.
/// Mirrors `librosa.effects.trim(top_db=40)`.
pub fn trim_silence(mono: &[f32], top_db: f32) -> (usize, usize) {
    let frame = 2048usize;
    let hop = 512usize;
    if mono.len() < frame {
        return (0, mono.len());
    }
    let n = 1 + (mono.len() - frame) / hop;
    let mut rms = vec![0.0f32; n];
    for (i, r) in rms.iter_mut().enumerate() {
        let s = &mono[i * hop..i * hop + frame];
        let ms = s.iter().map(|x| x * x).sum::<f32>() / frame as f32;
        *r = ms.sqrt();
    }
    let max = rms.iter().cloned().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return (0, mono.len());
    }
    let thresh = max * 10f32.powf(-top_db / 20.0);
    let first = rms.iter().position(|&r| r > thresh).unwrap_or(0);
    let last = rms.iter().rposition(|&r| r > thresh).unwrap_or(n - 1);
    let start = first * hop;
    let end = (last * hop + frame).min(mono.len());
    (start, end)
}

pub struct Features {
    /// 12 * n_frames, pitch-class major
    pub chroma: Vec<f32>,
    /// n_mels * n_frames, log-mel timbre (frame major is n_mels stride)
    pub mel: Vec<f32>,
    /// n_bins * n_frames, bin major, perceptually weighted dB
    pub power_db: Vec<f32>,
    pub n_bins: usize,
    pub n_frames: usize,
}

fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let n = signal.len();
    if n == 0 {
        return vec![0.0; pad * 2];
    }
    let mut out = Vec::with_capacity(n + 2 * pad);
    for j in 0..pad {
        let idx = (pad - j).min(n - 1);
        out.push(signal[idx]);
    }
    out.extend_from_slice(signal);
    for j in 0..pad {
        let idx = (n as isize - 2 - j as isize).max(0) as usize;
        out.push(signal[idx]);
    }
    out
}

fn a_weighting_db(freq: f32) -> f32 {
    let f = freq.max(1e-6);
    let f2 = f * f;
    let num = 12194.0f32.powi(2) * f2 * f2;
    let den = (f2 + 20.6f32.powi(2))
        * ((f2 + 107.7f32.powi(2)) * (f2 + 737.9f32.powi(2))).sqrt()
        * (f2 + 12194.0f32.powi(2));
    20.0 * (num / den).log10() - 2.0
}

fn build_chroma_filter(sr: u32, n_fft: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let mut filter = vec![0.0f32; 12 * n_bins];
    for k in 1..n_bins {
        let freq = k as f32 * sr as f32 / n_fft as f32;
        if freq <= 0.0 {
            continue;
        }
        let midi = 12.0 * (freq / 440.0).log2() + 69.0;
        let pc = midi.rem_euclid(12.0);
        for c in 0..12 {
            let mut d = (pc - c as f32).abs();
            if d > 6.0 {
                d = 12.0 - d;
            }
            let w = (-0.5 * (d / 0.5).powi(2)).exp();
            filter[c * n_bins + k] = w;
        }
    }
    filter
}

fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

fn build_mel_filter(sr: u32, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let fmin = 20.0f32;
    let fmax = 8000.0f32.min(sr as f32 / 2.0);
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let hz: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32))
        .collect();
    let mut filter = vec![0.0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let (lo, c, hi) = (hz[m], hz[m + 1], hz[m + 2]);
        for k in 0..n_bins {
            let f = k as f32 * sr as f32 / n_fft as f32;
            let w = if f >= lo && f <= c && c > lo {
                (f - lo) / (c - lo)
            } else if f > c && f <= hi && hi > c {
                (hi - f) / (hi - c)
            } else {
                0.0
            };
            filter[m * n_bins + k] = w;
        }
    }
    filter
}

/// Compute STFT-derived chroma, log-mel timbre and perceptual power (dB).
pub fn compute_features(mono: &[f32], sr: u32, n_fft: usize, hop: usize) -> Features {
    let n_bins = n_fft / 2 + 1;
    let n_frames = 1 + mono.len() / hop;
    let window = hann(n_fft);
    let padded = reflect_pad(mono, n_fft / 2);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let chroma_filter = build_chroma_filter(sr, n_fft);
    let mel_filter = build_mel_filter(sr, n_fft, N_MELS);
    let a_weight: Vec<f32> = (0..n_bins)
        .map(|k| {
            let f = k as f32 * sr as f32 / n_fft as f32;
            10f32.powf(a_weighting_db(f) / 10.0)
        })
        .collect();

    let mut chroma = vec![0.0f32; 12 * n_frames];
    let mut mel = vec![0.0f32; N_MELS * n_frames];
    let mut power_db = vec![0.0f32; n_bins * n_frames];

    let mut buf = vec![Complex::<f32>::new(0.0, 0.0); n_fft];
    for f in 0..n_frames {
        let start = f * hop;
        for i in 0..n_fft {
            let idx = (start + i).min(padded.len() - 1);
            buf[i] = Complex::new(padded[idx] * window[i], 0.0);
        }
        fft.process(&mut buf);

        // Power spectrogram and perceptual dB.
        for k in 0..n_bins {
            let p = buf[k].norm_sqr();
            power_db[k * n_frames + f] = 10.0 * (a_weight[k] * p).max(1e-10).log10();
        }
        // Chroma: filter @ power, then normalise frame by max.
        let mut frame = [0.0f32; 12];
        for (c, fv) in frame.iter_mut().enumerate() {
            let base = c * n_bins;
            let mut acc = 0.0f32;
            for k in 0..n_bins {
                let p = buf[k].norm_sqr();
                acc += chroma_filter[base + k] * p;
            }
            *fv = acc;
        }
        let max = frame.iter().cloned().fold(0.0f32, f32::max);
        let inv = if max > 0.0 { 1.0 / max } else { 0.0 };
        for (c, &v) in frame.iter().enumerate() {
            chroma[c * n_frames + f] = v * inv;
        }
        // Log-mel timbre.
        for m in 0..N_MELS {
            let base = m * n_bins;
            let mut acc = 0.0f32;
            for k in 0..n_bins {
                acc += mel_filter[base + k] * buf[k].norm_sqr();
            }
            mel[m * n_frames + f] = acc.max(1e-10).ln();
        }
    }

    Features {
        chroma,
        mel,
        power_db,
        n_bins,
        n_frames,
    }
}

/// Detect beats with the Ellis dynamic-programming tracker (pure Rust).
/// Returns (bpm, beat frame indices). Empty if detection failed.
pub fn detect_beats(path: &Path, sr: u32, hop: usize, n_frames: usize) -> (f32, Vec<usize>) {
    match beat_track_rs::analyze_file(path, &beat_track_rs::Params::default()) {
        Ok(res) => {
            let bpm = res.tempo_bpm as f32;
            let mut beats: Vec<usize> = res
                .beat_times_secs
                .iter()
                .map(|&t| (t * sr as f64 / hop as f64).round() as i64)
                .filter(|&i| i >= 0 && (i as usize) < n_frames)
                .map(|i| i as usize)
                .collect();
            beats.sort_unstable();
            beats.dedup();
            (bpm, beats)
        }
        Err(_) => (120.0, Vec::new()),
    }
}

/// Fallback uniform beat grid at the given tempo.
pub fn synth_beats(bpm: f32, sr: u32, hop: usize, n_frames: usize) -> Vec<usize> {
    let bpm = if bpm > 1.0 { bpm } else { 120.0 };
    let period = (60.0 / bpm * sr as f32 / hop as f32).round().max(1.0) as usize;
    (0..n_frames).step_by(period).collect()
}
