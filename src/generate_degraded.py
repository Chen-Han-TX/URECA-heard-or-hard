from pathlib import Path

import numpy as np
import soundfile as sf
import subprocess


CLEAN_DIR = Path("data/audio/clean")
NOISY_DIR = Path("data/audio/noisy")
HARD_DIR = Path("data/audio/hard")

SNR_DB = 5.0
SEED = 26060


def make_noisy(input_path: Path, output_path: Path, seed: int) -> None:
    audio, sample_rate = sf.read(input_path, dtype="float32")

    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio: {input_path}")

    rng = np.random.default_rng(seed)

    signal_power = np.mean(audio ** 2)

    noise_power = signal_power / (10 ** (SNR_DB / 10))

    noise = rng.normal(
        0.0,
        np.sqrt(noise_power),
        size=audio.shape,
    ).astype(np.float32)

    noisy = audio + noise

    peak = np.max(np.abs(noisy))

    if peak > 0.99:
        noisy = noisy / peak * 0.99

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(
        output_path,
        noisy,
        sample_rate,
        subtype="PCM_16",
    )

    actual_snr = 10 * np.log10(
        np.mean(audio ** 2) / np.mean(noise ** 2)
    )

    print(
        f"[NOISY] {input_path.name} "
        f"target={SNR_DB:.1f}dB "
        f"actual={actual_snr:.2f}dB"
    )


def make_hard(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-filter:a",
        "atempo=1.5,aresample=8000,aresample=16000",
        str(output_path),
    ]

    subprocess.run(
        command,
        check=True,
    )

    print(f"[HARD]  {input_path.name}")


def main() -> None:
    files = sorted(
        path
        for path in CLEAN_DIR.glob("*.wav")
        if path.stem.startswith(("e", "r"))
    )

    print(f"Found {len(files)} clean samples")

    for index, input_path in enumerate(files):
        noisy_path = NOISY_DIR / input_path.name
        hard_path = HARD_DIR / input_path.name

        make_noisy(
            input_path,
            noisy_path,
            seed=SEED + index,
        )

        make_hard(
            input_path,
            hard_path,
        )

    print()
    print("Done.")
    print(f"Noisy samples: {len(list(NOISY_DIR.glob('*.wav')))}")
    print(f"Hard samples : {len(list(HARD_DIR.glob('*.wav')))}")


if __name__ == "__main__":
    main()