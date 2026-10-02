"""Render a demo clip as a short, wide video and as a playback-bar image for the README.

GitHub strips <audio> and <video> tags from READMEs. It shows an inline player only for a
video uploaded as an attachment in its web editor, so the README uses one of two forms:
  - if docs/audio/players/attachments.json lists attachment URLs, those URLs (inline players);
  - otherwise the bar images, each linked to the hosted video, which plays on click.
Each video is the clip's waveform with a moving playhead and the audio itself, unchanged
apart from AAC encoding.

Usage: python experiments/make_players.py <utterance id> [<utterance id> ...]
"""
import os, subprocess, sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = "docs/audio/short", "docs/audio/players"
W, H = 1280, 220
KINDS = {"original": ("Original recording", "#2a78d6"), "reconstructed": ("Reconstructed from one gradient", "#d4472a")}


def frame(wav, sr, title, subtitle, color, path, button=False):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="#fcfcfb")
    left = 0.115 if button else 0.02
    ax = fig.add_axes([left, 0.14, 0.98 - left, 0.56], facecolor="#fcfcfb")
    if button:                                   # a play button, so the still image reads as a player
        bx = fig.add_axes([0.012, 0.17, 0.5 * H / W * 1.0, 0.5], facecolor="#fcfcfb")
        bx.set_xlim(-1, 1)
        bx.set_ylim(-1, 1)
        bx.set_aspect("equal")
        bx.axis("off")
        bx.add_patch(plt.Circle((0, 0), 0.95, color=color))
        bx.add_patch(plt.Polygon([[-0.28, -0.45], [-0.28, 0.45], [0.5, 0]], color="#fcfcfb"))
    n = 1200
    env = np.abs(wav[: len(wav) // n * n]).reshape(n, -1).max(1)
    env = env / (np.abs(wav).max() + 1e-9)
    x = np.linspace(0, len(wav) / sr, n)
    ax.fill_between(x, -env, env, color=color, linewidth=0)
    ax.set_xlim(0, len(wav) / sr)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    fig.text(0.02, 0.86, title, fontsize=15, fontweight="bold", color="#0b0b0b", va="center")
    fig.text(0.98, 0.86, subtitle, fontsize=11, color="#52514e", va="center", ha="right")
    fig.text(left, 0.05, "click to play" if button else "0 s", fontsize=9, color="#898781")
    fig.text(0.98, 0.05, f"{len(wav) / sr:.1f} s", fontsize=9, color="#898781", ha="right")
    fig.savefig(path, dpi=100, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for uid in sys.argv[1:]:
        for kind, (title, color) in KINDS.items():
            wav_path = os.path.join(SRC, f"{uid}_{kind}.wav")
            wav, sr = sf.read(wav_path, dtype="float32")
            dur = len(wav) / sr
            png = os.path.join(OUT, f"{uid}_{kind}.png")
            mp4 = os.path.join(OUT, f"{uid}_{kind}.mp4")
            frame(wav, sr, title, "LibriSpeech test-clean " + uid, color, png)
            frame(wav, sr, title, "LibriSpeech test-clean " + uid, color, os.path.join(OUT, f"{uid}_{kind}_bar.png"), button=True)
            x0, x1 = 0.02 * W, 0.98 * W           # the waveform axes span 2% to 98% of the width
            playhead = f"color=c=0x0b0b0b:s=3x{int(0.60 * H)}:r=30[bar];[0:v][bar]overlay=x='{x0}+({x1 - x0})*t/{dur}':y={int(0.15 * H)}:shortest=0[v]"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "30", "-i", png, "-i", wav_path,
                            "-filter_complex", playhead, "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "20", "-c:a", "aac", "-b:a", "128k", "-t", f"{dur:.3f}", "-movflags", "+faststart", mp4], check=True)
            os.remove(png)                       # the plain frame was only needed to build the video
            print(mp4, f"{os.path.getsize(mp4) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
