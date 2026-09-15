//! Waveform + marker renderer. Draws straight into an RGBA buffer using a
//! built-in 8x8 bitmap font (no font files, no external deps).

#![allow(clippy::needless_range_loop)]

use crate::assemble::Assembly;
use crate::report::Report;
use font8x8::UnicodeFonts;

#[derive(Clone)]
pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

const BG: [u8; 4] = [18, 18, 24, 255];
const TEXT: [u8; 4] = [232, 232, 238, 255];
const DIM: [u8; 4] = [150, 150, 160, 255];
const GRID: [u8; 4] = [42, 42, 52, 255];
const PLAYHEAD: [u8; 4] = [255, 66, 66, 255];

fn kind_color(kind: &str, loop_idx: usize) -> [u8; 4] {
    match kind {
        "intro" => [150, 150, 160, 255],
        "outro" => [255, 190, 90, 255],
        "tail" => [200, 150, 255, 255],
        "jump" => [90, 225, 200, 255],
        "play" => {
            if loop_idx.is_multiple_of(2) {
                [90, 170, 255, 255]
            } else {
                [130, 140, 255, 255]
            }
        }
        "loop" => [90, 170, 255, 255],
        _ => [200, 200, 200, 255],
    }
}

impl Canvas {
    pub fn new(w: usize, h: usize) -> Self {
        let mut px = vec![0u8; w * h * 4];
        for i in (0..px.len()).step_by(4) {
            px[i..i + 4].copy_from_slice(&BG);
        }
        Canvas { w, h, px }
    }

    #[inline]
    pub fn set(&mut self, x: i64, y: i64, c: [u8; 4]) {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return;
        }
        let i = ((y as usize) * self.w + x as usize) * 4;
        self.px[i..i + 4].copy_from_slice(&c);
    }

    pub fn fill_rect(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, c: [u8; 4]) {
        for y in y0..y1 {
            for x in x0..x1 {
                self.set(x, y, c);
            }
        }
    }

    pub fn vline(&mut self, x: i64, y0: i64, y1: i64, c: [u8; 4]) {
        for y in y0..y1 {
            self.set(x, y, c);
        }
    }

    pub fn text(&mut self, x: i64, y: i64, s: &str, c: [u8; 4], scale: i64) {
        let mut cx = x;
        for ch in s.chars() {
            if let Some(glyph) = font8x8::BASIC_FONTS.get(ch) {
                for (row, bits) in glyph.iter().enumerate() {
                    for col in 0..8i64 {
                        if (bits >> col) & 1 == 1 {
                            self.fill_rect(
                                cx + col * scale,
                                y + row as i64 * scale,
                                cx + (col + 1) * scale,
                                y + (row as i64 + 1) * scale,
                                c,
                            );
                        }
                    }
                }
            }
            cx += 8 * scale + scale; // glyph + 1px gap
        }
    }
}

fn fmt_time(secs: f64) -> String {
    let s = secs.max(0.0);
    let m = (s / 60.0).floor() as u64;
    let r = s - m as f64 * 60.0;
    format!("{m}:{r:05.2}")
}

/// Build the static background (waveform, regions, markers, text) once.
pub fn background(assembly: &Assembly, report: &Report, w: usize, h: usize) -> Canvas {
    let mut c = Canvas::new(w, h);
    let n = assembly.frames().max(1);
    let mono = mono_of(assembly);

    let top = (h as f64 * 0.18) as usize;
    let bottom = h - (h as f64 * 0.14) as usize;
    let center = (top + bottom) / 2;
    let half = (bottom - top) as f64 / 2.0;

    // Per-column colour from the segment that owns that output time.
    let mut col_color = vec![[150u8, 150, 150, 255]; w];
    let mut loop_idx = 0usize;
    for seg in &assembly.segments {
        let color = kind_color(seg.kind, if seg.kind == "loop" { loop_idx } else { 0 });
        if seg.kind == "loop" {
            loop_idx += 1;
        }
        let x0 = (seg.out_start as f64 / n as f64 * w as f64) as usize;
        let x1 = (seg.out_end as f64 / n as f64 * w as f64).round() as usize;
        for x in x0..x1.min(w) {
            col_color[x] = color;
        }
    }

    // Region band across the top.
    for (x, color) in col_color.iter().enumerate() {
        for y in 0..6 {
            c.set(x as i64, y, *color);
        }
    }

    // Waveform.
    for x in 0..w {
        let s0 = x * n / w;
        let s1 = (((x + 1) * n / w).max(s0 + 1)).min(n);
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for &v in &mono[s0..s1] {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        if lo > hi {
            lo = 0.0;
            hi = 0.0;
        }
        let y_hi = center as f64 - hi as f64 * half;
        let y_lo = center as f64 - lo as f64 * half;
        c.vline(x as i64, y_hi as i64, (y_lo as i64 + 1).max(y_hi as i64 + 1), col_color[x]);
    }

    // Centre grid line.
    c.vline(0, center as i64, center as i64 + 1, GRID);

    // Segment markers.
    loop_idx = 0;
    for seg in &assembly.segments {
        let color = kind_color(seg.kind, if seg.kind == "loop" { loop_idx } else { 0 });
        if seg.kind == "loop" {
            loop_idx += 1;
        }
        let x = (seg.out_start as f64 / n as f64 * w as f64) as i64;
        for y in 0..h as i64 {
            if y % 6 < 3 {
                c.set(x, y, color);
            }
        }
        c.fill_rect(x, 8, x + 8, 16, color);
    }

    // Header text.
    let name = std::path::Path::new(&report.input)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("track");
    let title = format!(
        "REMiX  {}  |  target {}  got {}  |  {} BPM",
        name,
        fmt_time(report.target_secs),
        fmt_time(report.result_secs),
        report.bpm.round() as i64
    );
    c.text(8, 22, &title, TEXT, 2);
    let sub = format!(
        "loop {} - {}  ({} full loops)   trim {} - {}",
        fmt_time(report.loop_start_secs),
        fmt_time(report.loop_end_secs),
        report.loop_iterations,
        fmt_time(report.trimmed_start_secs),
        fmt_time(report.trimmed_end_secs),
    );
    c.text(8, 44, &sub, DIM, 1);

    // Legend at the bottom.
    let y = (h - 26) as i64;
    let mut x = 8i64;
    let items: [(&str, [u8; 4]); 4] = [
        ("intro", kind_color("intro", 0)),
        ("play", kind_color("play", 0)),
        ("jump", kind_color("jump", 0)),
        ("outro", kind_color("outro", 0)),
    ];
    for (label, color) in items {
        c.fill_rect(x, y, x + 12, y + 12, color);
        c.text(x + 18, y + 2, label, TEXT, 1);
        x += 18 + (label.len() as i64) * 9 + 22;
    }

    c
}

pub fn draw_playhead(canvas: &mut Canvas, frac: f64) {
    let x = (frac.clamp(0.0, 1.0) * canvas.w as f64) as i64;
    canvas.vline(x, 0, canvas.h as i64, PLAYHEAD);
    canvas.vline(x - 1, 0, canvas.h as i64, [120, 20, 20, 255]);
}

pub fn mono_of(assembly: &Assembly) -> Vec<f32> {
    let n = assembly.frames();
    let nch = assembly.channels.len().max(1) as f32;
    let mut out = vec![0.0f32; n];
    for ch in &assembly.channels {
        for (o, s) in out.iter_mut().zip(ch.iter()) {
            *o += *s;
        }
    }
    for v in out.iter_mut() {
        *v /= nch;
    }
    out
}
