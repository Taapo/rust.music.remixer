# rust.music.remixer

A single, dependency-free Rust binary that remixes / loops any music track to a
**target length** — the open-source equivalent of Adobe Audition / Premiere
"Remix" and DaVinci Resolve's music remix, plus a **waveform + markers MP4** so
you can scrub and hear exactly what was done.

It analyses the track, finds the most seamless loop points, and assembles
`intro → loop × N → outro` to land on (almost) the exact length you asked for,
accounting for a dead tail at the end of the file.

```
remix in.mp3 --length 1:30 --out out.wav --mp4 out.mp4
```

## Why

Finding a track that is *exactly* the length you need is a daily editing chore.
The commercial tools that solve it are closed. This is a small, auditable,
offline tool with no Python, no ffmpeg, no models, and no external files at
runtime.

## Features

- **Target length**: `--length 90`, `--length 1:30`, or `--length 0:01:30`.
- **Smart loop finding** (ported from [PyMusicLooper](https://github.com/arkrow/PyMusicLooper), MIT):
  beat tracking, chroma/loudness similarity, cosine sub-sequence scoring, longest-good-loop preference.
- **Tail handling**: leading/trailing silence is trimmed (`top_db = 40`) before analysis.
- **Zero-crossing snapping** + short crossfades at every seam to avoid clicks.
- **Outputs**
  - 32-bit float WAV (default), or 16/24/32-bit PCM (`--wav-format`).
  - MP3 with a selectable bitrate (`--mp3`, `--mp3-bitrate`).
  - **MP4** waveform video with colour-coded regions, markers and a moving
    playhead (`--mp4`), H.264 + AAC-LC, fast-start.
- **JSON timeline report**: every segment with its output time and source time.

## Install / build

Requires a Rust toolchain (and, on Windows, the MSVC C++ build tools — OpenH264
and LAME are compiled from source; `nasm` on `PATH` speeds up OpenH264 but is
optional).

```sh
cargo build --release
# target/release/remix(.exe)
```

## Usage

```
remix <INPUT> --length <LEN> [options]

  --out <FILE.wav>        WAV output (default: <input>.remix.wav)
  --wav-format <FMT>      f32 (default) | s16 | s24 | s32
  --mp3 <FILE.mp3>        also write MP3
  --mp3-bitrate <KBPS>    MP3 CBR bitrate, default 320
  --mp4 <FILE.mp4>        also write the waveform/markers video
  --fps <N>               video frame rate, default 50
  --size <WxH>            video size, default 1920x1080
  --no-outro              end by fading the last loop instead of the outro
  --report <FILE.json>    timeline report path
  -q, --quiet             less output
```

Examples:

```sh
# 90-second WAV master + 320k MP3
remix song.flac --length 1:30 --out song90.wav --mp3 song90.mp3 --mp3-bitrate 320

# 45-second remix with the review video
remix song.mp3 --length 45 --mp4 song45.mp4
```

## How it works

1. Decode (Symphonia) → mono for analysis, full channels for output.
2. Trim leading/trailing silence.
3. STFT (rustfft) → chroma + perceptually-weighted power (dB).
4. Beat tracking (Ellis dynamic programming).
5. Candidate loop points from note-distance + loudness difference, scored by
   cosine similarity of the surrounding beats; prefer the longest among the
   top-scoring pairs.
6. Snap loop edges to rising zero crossings.
7. Assemble `intro + loops + partial-loop filler + outro` to the target length,
   crossfading seams.
8. Encode WAV / MP3 / MP4, and write the JSON report.

## Video output

The MP4 is a full-track static waveform of the **remix** so every event is
visible at once:

- Grey `intro`, alternating blue/teal `loop` iterations, purple filler `tail`, amber `outro`.
- A vertical marker at each seam and a legend.
- A red playhead sweeping in sync with the audio.

## Known limitations

- If the track has no detectable loop, it falls back to looping the whole track.
- Mixing is a straight concatenation with crossfades — no generative
  rearrangement (that's the Remixatron "infinite jukebox" style, a different tool).
- AAC at 320 kbps.

## Credits / licenses

This project is MIT. It leans on the algorithm design of
[PyMusicLooper](https://github.com/arkrow/PyMusicLooper) (MIT) and the Ellis
beat-tracking method.

Dependencies and their licenses — note the two non-permissive ones if you plan
to redistribute:

| Crate | License |
| --- | --- |
| symphonia | MPL-2.0 |
| rustfft | MIT/Apache-2.0 |
| beat-track-rs | MIT/Apache-2.0 |
| hound | Apache-2.0 |
| rusty_aac | Apache-2.0 |
| openh264 | BSD-2-Clause (Cisco) |
| muxide | MIT/Apache-2.0 |
| **mp3lame-encoder / LAME** | **LGPL-2.0** |
| clap, serde, anyhow, font8x8 | MIT/Apache-2.0 |
