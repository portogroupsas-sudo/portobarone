#!/usr/bin/env python3
"""Limpia un audio y aísla la voz. Todo el procesado se hace a 48 kHz.

Cadena: ffmpeg (decodificar + quitar zumbido) -> WPE -> DeepFilterNet3
        -> noisereduce suave (perfil de frames silenciosos) -> loudnorm -16 LUFS.

Uso:
    python limpiar_voz.py entrada.ext salida.wav [opciones]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000  # Nunca bajar a 16 kHz.


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def leer(path):
    x, sr = sf.read(str(path), dtype="float32")
    if sr != SR:
        sys.exit(f"Error: {path} está a {sr} Hz, se esperaba {SR} Hz")
    return x if x.ndim == 1 else x.mean(axis=1)


def escribir(path, x):
    sf.write(str(path), x.astype(np.float32), SR, subtype="FLOAT")


def decodificar_sin_zumbido(entrada, salida, hum):
    notches = ",".join(
        f"equalizer=f={hum * k}:t=q:w=30:g={g}"
        for k, g in zip(range(1, 6), (-30, -25, -20, -15, -12))
    )
    af = f"highpass=f={hum}:poles=2,{notches}"
    run(["ffmpeg", "-y", "-i", str(entrada), "-vn", "-ac", "1", "-ar", str(SR),
         "-af", af, "-c:a", "pcm_f32le", str(salida)])


def aplicar_wpe(entrada, salida, taps):
    from nara_wpe.utils import istft, stft
    from nara_wpe.wpe import wpe

    y = leer(entrada)
    size, shift = 2048, 512  # ~43 ms / ~10.7 ms a 48 kHz
    Y = stft(y[np.newaxis, :], size=size, shift=shift).transpose(2, 0, 1)
    Z = wpe(Y, taps=taps, delay=3, iterations=3, statistics_mode="full")
    z = istft(Z.transpose(1, 2, 0), size=size, shift=shift)[0, : len(y)]
    escribir(salida, z)


def aplicar_deepfilternet3(entrada, salida):
    from df.enhance import enhance, init_df, load_audio, save_audio

    model, df_state, _ = init_df(model_base_dir="DeepFilterNet3")
    if df_state.sr() != SR:
        sys.exit(f"Error: DeepFilterNet trabaja a {df_state.sr()} Hz, se esperaba {SR}")
    audio, _ = load_audio(str(entrada), sr=SR)
    save_audio(str(salida), enhance(model, df_state, audio), SR)


def perfil_ruido(x, frame_s=0.05, fraccion=0.10, minimo_s=0.5):
    frame = int(frame_s * SR)
    n = len(x) // frame
    if n == 0:
        return x
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    k = min(n, max(int(n * fraccion), int(minimo_s / frame_s)))
    idx = np.sort(np.argsort(rms)[:k])
    return np.concatenate([x[i * frame:(i + 1) * frame] for i in idx])


def aplicar_noisereduce(entrada, salida, prop):
    import noisereduce as nr

    x = leer(entrada)
    limpio = nr.reduce_noise(y=x, sr=SR, y_noise=perfil_ruido(x),
                             stationary=True, prop_decrease=prop)
    escribir(salida, limpio)


def normalizar(entrada, salida, lufs, tp=-1.5, lra=11):
    base = f"loudnorm=I={lufs}:TP={tp}:LRA={lra}"
    medida = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(entrada), "-af",
         f"{base}:print_format=json", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stderr
    m = json.loads(re.findall(r"\{[^{}]*\}", medida)[-1])
    af = (f"{base}:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
          f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
          f":offset={m['target_offset']}:linear=true")
    # loudnorm sobremuestrea a 192 kHz internamente: forzar 48 kHz a la salida.
    run(["ffmpeg", "-y", "-i", str(entrada), "-af", af, "-ar", str(SR),
         "-c:a", "pcm_s24le", str(salida)])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("entrada", type=Path)
    p.add_argument("salida", type=Path)
    p.add_argument("--hum", type=int, default=60, help="frecuencia de red (60 o 50)")
    p.add_argument("--sin-wpe", action="store_true", help="saltar WPE")
    p.add_argument("--wpe-taps", type=int, default=10)
    p.add_argument("--nr", type=float, default=0.5, help="prop_decrease de noisereduce")
    p.add_argument("--lufs", type=float, default=-16.0)
    p.add_argument("--guardar-intermedios", action="store_true")
    a = p.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("Error: ffmpeg no está instalado")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        etapas = [tmp / f for f in ("01_sin_zumbido.wav", "02_wpe.wav",
                                    "03_dfn3.wav", "04_nr.wav")]

        print("1/5 Decodificando a 48 kHz y quitando zumbido...")
        decodificar_sin_zumbido(a.entrada, etapas[0], a.hum)

        if a.sin_wpe:
            print("2/5 WPE omitido")
            shutil.copy(etapas[0], etapas[1])
        else:
            print("2/5 WPE (eco/reverberación)...")
            aplicar_wpe(etapas[0], etapas[1], a.wpe_taps)

        print("3/5 DeepFilterNet3 (aislar voz)...")
        aplicar_deepfilternet3(etapas[1], etapas[2])

        print("4/5 noisereduce suave (perfil de frames silenciosos)...")
        aplicar_noisereduce(etapas[2], etapas[3], a.nr)

        print(f"5/5 Normalizando a {a.lufs} LUFS...")
        normalizar(etapas[3], a.salida, a.lufs)

        if a.guardar_intermedios:
            destino = a.salida.with_name(a.salida.stem + "_etapas")
            destino.mkdir(parents=True, exist_ok=True)
            for e in etapas:
                shutil.copy(e, destino / e.name)
            print(f"Intermedios guardados en {destino}")

    sr = run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate",
              "-of", "default=nw=1:nk=1", str(a.salida)]).stdout.strip()
    print(f"Listo: {a.salida} ({sr} Hz)")
    if sr != str(SR):
        sys.exit(f"Error: la salida está a {sr} Hz, no a {SR} Hz")


if __name__ == "__main__":
    main()
