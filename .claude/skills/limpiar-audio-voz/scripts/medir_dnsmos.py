#!/usr/bin/env python3
"""Estima la calidad de voz con DNSMOS P.835 (SIG, BAK, OVRL; escala 1-5).

Sólo mide: analiza una copia remuestreada a 16 kHz (lo que exige el modelo)
con el nivel igualado; el audio original no se modifica.

Uso:
    python medir_dnsmos.py --modelo sig_bak_ovr.onnx a.wav [b.wav ...]
"""
import argparse
from math import gcd

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

SR_MODELO = 16000
VENTANA = int(9.01 * SR_MODELO)
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])


def preparar(path):
    x, sr = sf.read(str(path))
    x = x.mean(axis=1) if x.ndim > 1 else x
    g = gcd(sr, SR_MODELO)
    x = resample_poly(x, SR_MODELO // g, sr // g)
    # Igualar nivel (-20 dBFS en la mitad más fuerte de los frames de 20 ms):
    # DNSMOS depende del volumen y sin esto las comparaciones engañan.
    fr = x[: len(x) // 320 * 320].reshape(-1, 320)
    e = (fr ** 2).mean(axis=1)
    return (x * 0.1 / np.sqrt(e[e >= np.median(e)].mean())).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", required=True, help="ruta a sig_bak_ovr.onnx")
    p.add_argument("audios", nargs="+")
    a = p.parse_args()

    sess = ort.InferenceSession(a.modelo)
    entrada = sess.get_inputs()[0].name
    for path in a.audios:
        x = preparar(path)
        if len(x) < VENTANA:
            x = np.pad(x, (0, VENTANA - len(x)))
        r = []
        for i in range(0, len(x) - VENTANA + 1, SR_MODELO):
            sig, bak, ovr = sess.run(None, {entrada: x[i:i + VENTANA][None]})[0][0]
            r.append([P_SIG(sig), P_BAK(bak), P_OVR(ovr)])
        s, b, o = np.mean(r, axis=0)
        print(f"{path}: SIG {s:.2f}  BAK {b:.2f}  OVRL {o:.2f}")


if __name__ == "__main__":
    main()
