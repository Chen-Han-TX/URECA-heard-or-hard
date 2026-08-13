from pathlib import Path

import numpy as np
import soundfile as sf


INPUT = Path("data/audio/clean/test.wav")
OUTPUT = Path("data/audio/noisy/test_5db.wav")

SNR_DB = 5.0
SEED = 26060


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    audio, sample_rate = sf.read(INPUT, dtype="float32")

    if audio.ndim != 1:
        raise ValueError("Expected mono audio")

    rng = np.random.default_rng(SEED)

    signal_power = np.mean(audio ** 2)

    noise_power = signal_power / (10 ** (SNR_DB / 10))

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=audio.shape,
    ).astype(np.float32)

    noisy = audio + noise

    # Prevent clipping.
    peak = np.max(np.abs(noisy))
    if peak > 0.99:
        noisy = noisy / peak * 0.99

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    sf.write(
        OUTPUT,
        noisy,
        sample_rate,
        subtype="PCM_16",
    )

    actual_signal_power = np.mean(audio ** 2)
    actual_noise_power = np.mean(noise ** 2)

    actual_snr = 10 * np.log10(
        actual_signal_power / actual_noise_power
    )

    print(f"Saved: {OUTPUT}")
    print(f"Target SNR: {SNR_DB:.2f} dB")
    print(f"Actual SNR: {actual_snr:.2f} dB")


if __name__ == "__main__":
    main()