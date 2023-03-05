import argparse
import numpy as np
import librosa
import soundfile as sf


def phase_vocoder(
    D: np.array,
    rate: float,
    ) -> np.array:

    n_fft = 2 * (D.shape[-2] - 1)
    hop_length = int(n_fft // 4)

    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)

    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros(shape=shape, dtype=D.dtype)

    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[-2])

    phase_acc = np.angle(D[:, 0])

    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D = np.pad(D, padding, mode="constant")

    for t, step in enumerate(time_steps):
        start = int(step)
        end = start + 2
        columns = D[:, start:end]

        alpha = step - start

        mag = (1.0 - alpha) * np.abs(columns[:, 0]) + alpha * np.abs(columns[:, 1])

        d_stretch[:, t] = np.cos(phase_acc) + 1j * np.sin(phase_acc) * mag

        dphase = np.angle(columns[:, 1]) - np.angle(columns[:, 0]) - phi_advance
        dphase = np.mod(dphase + np.pi, 2 * np.pi) - np.pi

        phase_acc += phi_advance + dphase

    return d_stretch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase Vocoder")
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("time_stretch_ratio", type=float)
    args = parser.parse_args()
    y, sr = librosa.load(args.input)
    D = librosa.stft(y)
    D_stretched = phase_vocoder(D, rate=args.time_stretch_ratio)
    y_stretched = librosa.istft(D_stretched)
    soundfile.write(args.output, y_stretched, sr)
