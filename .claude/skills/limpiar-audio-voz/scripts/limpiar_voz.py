#!/usr/bin/env python3
"""Limpia un audio y aísla la voz. Todo el procesado se hace a 48 kHz.

Cadena: ffmpeg (decodificar + quitar zumbido, conservando los canales)
        -> WPE multicanal -> DeepFilterNet3 (+ post-filtro)
        -> noisereduce (perfil de frames silenciosos)
        -> mezcla en seco de la señal tras WPE (quita el sonido "robótico")
        -> EQ/compresión/de-esser de podcast
        -> expander por energía (atenúa voces lejanas/de fondo)
        -> limitador + loudnorm -16 LUFS.

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
import types
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000  # Nunca bajar a 16 kHz.

# EQ de podcast: limpia graves, quita "barro" en 280 Hz, presencia en 3.5 kHz,
# un poco de aire, compresión suave y de-esser.
PODCAST = ("highpass=f=80:poles=2,"
           "equalizer=f=280:t=q:w=1.2:g=-2,"
           "equalizer=f=3500:t=q:w=1.0:g=2.5,"
           "highshelf=f=9000:g=1.5,"
           "acompressor=threshold=-22dB:ratio=2.5:attack=10:release=150:knee=6:makeup=3,"
           "deesser=i=0.3:f=0.55")

# Expander: umbral relativo al nivel local de la voz principal.
EXPANDER = {
    "suave": dict(bajo_pico=10, ratio=3, rango=15, hold=100, release=200, lookahead=20),
    "fuerte": dict(bajo_pico=5, ratio=6, rango=35, hold=50, release=120, lookahead=15),
}


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def leer(path, mono=True):
    x, sr = sf.read(str(path), dtype="float32")
    if sr != SR:
        sys.exit(f"Error: {path} está a {sr} Hz, se esperaba {SR} Hz")
    return x.mean(axis=1) if mono and x.ndim > 1 else x


def escribir(path, x):
    sf.write(str(path), np.asarray(x, dtype=np.float32), SR, subtype="FLOAT")


def decodificar_sin_zumbido(entrada, salida, hum, mono):
    notches = ",".join(
        f"equalizer=f={hum * k}:t=q:w=30:g={g}"
        for k, g in zip(range(1, 6), (-30, -25, -20, -15, -12))
    )
    af = f"highpass=f={hum}:poles=2,{notches}"
    canales = ["-ac", "1"] if mono else []
    run(["ffmpeg", "-y", "-i", str(entrada), "-vn", *canales, "-ar", str(SR),
         "-af", af, "-c:a", "pcm_f32le", str(salida)])


def aplicar_wpe(entrada, salida, taps):
    """WPE con todos los canales disponibles (estéreo quita mucho más eco)."""
    from nara_wpe.utils import istft, stft
    from nara_wpe.wpe import wpe

    y = leer(entrada, mono=False).astype(np.float64)
    y = y[np.newaxis, :] if y.ndim == 1 else y.T  # (C, N)
    size, shift = 2048, 512  # ~43 ms / ~10.7 ms a 48 kHz
    Y = stft(y, size=size, shift=shift).transpose(2, 0, 1)  # (F, C, T)
    Z = wpe(Y, taps=taps, delay=3, iterations=5, statistics_mode="full")
    z = istft(Z.transpose(1, 2, 0), size=size, shift=shift)[:, : y.shape[1]]
    escribir(salida, z.mean(axis=0))


def aplicar_deepfilternet3(entrada, salida, post_filter):
    # Compatibilidad con torchaudio >= 2.2 (ya no existe torchaudio.backend).
    try:
        import torchaudio.backend.common  # noqa: F401
    except ImportError:
        comun = types.ModuleType("torchaudio.backend.common")
        comun.AudioMetaData = type("AudioMetaData", (), {})
        sys.modules["torchaudio.backend"] = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend.common"] = comun
    import torch
    from df.enhance import enhance, init_df

    model, df_state, _ = init_df(model_base_dir="DeepFilterNet3", post_filter=post_filter)
    if df_state.sr() != SR:
        sys.exit(f"Error: DeepFilterNet trabaja a {df_state.sr()} Hz, se esperaba {SR}")
    # Se lee con soundfile: torchaudio >= 2.9 ya no trae info()/load().
    audio = torch.from_numpy(leer(entrada))[None]
    escribir(salida, enhance(model, df_state, audio).squeeze(0).numpy())


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


def mezclar_seco(entrada, seco, salida, db):
    """Devuelve un poco de la voz sin red neuronal: menos timbre robótico."""
    x, s = leer(entrada), leer(seco)
    n = min(len(x), len(s))
    escribir(salida, x[:n] + s[:n] * 10 ** (db / 20))


def aplicar_podcast(entrada, salida):
    run(["ffmpeg", "-y", "-i", str(entrada), "-af", PODCAST, "-ar", str(SR),
         "-c:a", "pcm_f32le", str(salida)])


def aplicar_expander(entrada, salida, bajo_pico, ratio, rango, hold, release,
                     lookahead, attack=5.0, rodilla=6.0, ventana=4.0, caida_max=15.0):
    """Expander descendente: atenúa lo que queda muy por debajo de la voz principal.

    El umbral sigue el nivel local (p90 en ~4 s) de la voz principal, así una voz
    principal que habla más bajo no se castiga. Hold causal + release largos y
    lookahead para no cortar ataques ni finales de palabra.
    """
    from scipy.ndimage import maximum_filter1d, percentile_filter
    from scipy.signal import butter, sosfiltfilt

    x = leer(entrada).astype(np.float64)
    det = sosfiltfilt(butter(4, [150, 4000], "bandpass", fs=SR, output="sos"), x)
    w = int(0.010 * SR)
    pot = np.convolve(det ** 2, np.ones(w) / w, mode="same")
    env = 10 * np.log10(pot + 1e-12)

    f = int(0.02 * SR)
    n = len(env) // f
    frames = 10 * np.log10(pot[: n * f].reshape(n, f).mean(axis=1) + 1e-12)
    local = percentile_filter(frames, 90, size=int(ventana / 0.02), mode="nearest")
    local = np.maximum(local, np.percentile(frames, 95) - caida_max)
    thr = np.interp(np.arange(len(env)), np.arange(n) * f + f / 2, local - bajo_pico)

    d = env - thr
    k = rodilla / 2
    g = np.where(d >= k, 0.0,
                 np.where(d <= -k, d * (ratio - 1),
                          (ratio - 1) * -((d - k) ** 2) / (4 * k)))
    g = np.maximum(g, -rango)

    h = int(hold / 1000 * SR)
    g = maximum_filter1d(g, size=h + 1)
    g = np.concatenate([np.full(h // 2, g[0]), g[: len(g) - h // 2]])

    ca = np.exp(-1 / (attack / 1000 * SR))
    cr = np.exp(-1 / (release / 1000 * SR))
    s = np.empty_like(g)
    y = g[0]
    for i, v in enumerate(g):
        c = ca if v > y else cr
        y = c * y + (1 - c) * v
        s[i] = y

    la = int(lookahead / 1000 * SR)
    s = np.concatenate([s[la:], np.full(la, s[-1])])
    escribir(salida, x * 10 ** (s / 20))


def medir_loudnorm(entrada, base):
    medida = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(entrada), "-af",
         f"{base}:print_format=json", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stderr
    return json.loads(re.findall(r"\{[^{}]*\}", medida)[-1])


def normalizar(entrada, salida, lufs, tp=-1.5, lra=11):
    base = f"loudnorm=I={lufs}:TP={tp}:LRA={lra}"
    # Limitador previo: sin él, el pico verdadero impide llegar al objetivo
    # (se queda ~1 dB corto). Sube hasta el objetivo y limita los picos.
    m = medir_loudnorm(entrada, base)
    ganancia = lufs - float(m["input_i"])
    limitado = Path(salida).with_suffix(".lim.wav")
    run(["ffmpeg", "-y", "-i", str(entrada), "-af",
         f"volume={ganancia:.2f}dB,alimiter=limit={10 ** ((tp - 0.5) / 20):.4f}"
         ":attack=2:release=60:level=disabled",
         "-ar", str(SR), "-c:a", "pcm_f32le", str(limitado)])
    m = medir_loudnorm(limitado, base)
    af = (f"{base}:measured_I={m['input_i']}:measured_TP={m['input_tp']}"
          f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
          f":offset={m['target_offset']}:linear=true")
    # loudnorm sobremuestrea a 192 kHz internamente: forzar 48 kHz a la salida.
    run(["ffmpeg", "-y", "-i", str(limitado), "-af", af, "-ar", str(SR),
         "-c:a", "pcm_s24le", str(salida)])
    limitado.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("entrada", type=Path)
    p.add_argument("salida", type=Path)
    p.add_argument("--hum", type=int, default=60, help="frecuencia de red (60 o 50)")
    p.add_argument("--sin-wpe", action="store_true", help="saltar WPE")
    p.add_argument("--wpe-taps", type=int, default=10)
    p.add_argument("--wpe-mono", action="store_true",
                   help="WPE sobre la mezcla mono aunque la entrada sea estéreo")
    p.add_argument("--sin-postfiltro", action="store_true",
                   help="DeepFilterNet3 sin post-filtro (menos supresión)")
    p.add_argument("--nr", type=float, default=0.6, help="prop_decrease de noisereduce")
    p.add_argument("--seco", type=float, default=-20.0,
                   help="dB de señal tras WPE mezclada de vuelta (-99 = nada)")
    p.add_argument("--sin-podcast", action="store_true", help="sin EQ/compresión")
    p.add_argument("--expander", choices=["fuerte", "suave", "no"], default="fuerte")
    p.add_argument("--lufs", type=float, default=-16.0)
    p.add_argument("--guardar-intermedios", action="store_true")
    a = p.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("Error: ffmpeg no está instalado")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        e = {k: tmp / f"{k}.wav" for k in (
            "01_sin_zumbido", "02_wpe", "03_dfn3", "04_nr", "05_seco",
            "06_podcast", "07_expander")}

        print("1/8 Decodificando a 48 kHz y quitando zumbido...")
        decodificar_sin_zumbido(a.entrada, e["01_sin_zumbido"], a.hum, a.wpe_mono)

        if a.sin_wpe:
            print("2/8 WPE omitido")
            escribir(e["02_wpe"], leer(e["01_sin_zumbido"]))
        else:
            print("2/8 WPE multicanal (eco/reverberación)...")
            aplicar_wpe(e["01_sin_zumbido"], e["02_wpe"], a.wpe_taps)

        print("3/8 DeepFilterNet3 (aislar voz)...")
        aplicar_deepfilternet3(e["02_wpe"], e["03_dfn3"], not a.sin_postfiltro)

        print("4/8 noisereduce (perfil de frames silenciosos)...")
        aplicar_noisereduce(e["03_dfn3"], e["04_nr"], a.nr)

        print(f"5/8 Mezcla en seco a {a.seco} dB...")
        mezclar_seco(e["04_nr"], e["02_wpe"], e["05_seco"], a.seco)

        if a.sin_podcast:
            print("6/8 EQ de podcast omitida")
            shutil.copy(e["05_seco"], e["06_podcast"])
        else:
            print("6/8 EQ, compresión y de-esser de podcast...")
            aplicar_podcast(e["05_seco"], e["06_podcast"])

        if a.expander == "no":
            print("7/8 Expander omitido")
            shutil.copy(e["06_podcast"], e["07_expander"])
        else:
            print(f"7/8 Expander ({a.expander}) para voces de fondo...")
            aplicar_expander(e["06_podcast"], e["07_expander"], **EXPANDER[a.expander])

        print(f"8/8 Limitador y normalización a {a.lufs} LUFS...")
        normalizar(e["07_expander"], a.salida, a.lufs)

        if a.guardar_intermedios:
            destino = a.salida.with_name(a.salida.stem + "_etapas")
            destino.mkdir(parents=True, exist_ok=True)
            for f in e.values():
                shutil.copy(f, destino / f.name)
            print(f"Intermedios guardados en {destino}")

    sr = run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate",
              "-of", "default=nw=1:nk=1", str(a.salida)]).stdout.strip()
    print(f"Listo: {a.salida} ({sr} Hz)")
    if sr != str(SR):
        sys.exit(f"Error: la salida está a {sr} Hz, no a {SR} Hz")


if __name__ == "__main__":
    main()
