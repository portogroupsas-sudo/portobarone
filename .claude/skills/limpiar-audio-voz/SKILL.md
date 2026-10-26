---
name: limpiar-audio-voz
description: Limpia grabaciones y aísla la voz trabajando siempre a 48 kHz (nunca baja a 16 kHz). Cadena con ffmpeg, eliminación de zumbido de 60 Hz, WPE para el eco/reverberación, DeepFilterNet3 para aislar la voz, noisereduce suave con perfil de los frames más silenciosos y normalización a -16 LUFS. Úsala cuando pidan limpiar un audio, quitar ruido, eco o zumbido, aislar o mejorar la voz de una grabación, entrevista, podcast o vídeo.
---

# Limpiar audio y aislar la voz

Cadena de procesado para dejar una voz limpia, sin zumbido, con poco eco y a
volumen estándar de podcast/redes (-16 LUFS).

## Regla innegociable: 48 kHz en todo momento

- Todo el procesado se hace a **48 000 Hz**. **Nunca** remuestrear a 16 kHz
  (ni a 22.05/24/44.1 kHz) en ningún paso intermedio ni en la salida.
- DeepFilterNet3 trabaja nativamente a 48 kHz: no hace falta (ni se debe)
  bajar la frecuencia para él.
- `loudnorm` de ffmpeg sobremuestrea internamente a 192 kHz; por eso la salida
  **siempre** lleva `-ar 48000` explícito.
- Formato intermedio: WAV mono `pcm_f32le` (float 32) para no perder rango ni
  provocar clipping entre etapas.
- Al terminar, verifica con `ffprobe` que la salida es 48000 Hz.

## Orden de la cadena (y por qué)

1. **Decodificar con ffmpeg → WAV mono float 48 kHz.** Acepta cualquier entrada
   (mp3, m4a, mp4, mov, wav…).
2. **Quitar zumbido de 60 Hz** (filtros lineales de ffmpeg): notch en 60 Hz y
   armónicos (120, 180, 240, 300 Hz) + pasa-altos suave a 60 Hz para el rumble.
   Si la grabación es de una red de 50 Hz (Europa), cambiar a 50/100/150/200/250.
3. **WPE (nara_wpe) para el eco/reverberación.** Va antes que los denoisers
   porque WPE es un predictor lineal y necesita la señal sin procesar de forma
   no lineal.
4. **DeepFilterNet3 para aislar la voz.** Modelo `DeepFilterNet3`, a 48 kHz.
5. **noisereduce suave** con perfil de ruido tomado de los **frames más
   silenciosos** (no de los primeros segundos): `stationary=True`,
   `prop_decrease≈0.5`. Sólo pule el ruido residual; si se sube mucho aparecen
   artefactos "acuáticos".
6. **Normalizar a -16 LUFS** con `loudnorm` en dos pasadas (I=-16, TP=-1.5,
   LRA=11) y salida a 48 kHz.

## Instalación

```bash
# ffmpeg (sistema)
sudo apt-get install -y ffmpeg        # macOS: brew install ffmpeg

# Python
pip install numpy soundfile nara_wpe noisereduce deepfilternet
```

Nota DeepFilterNet: con torchaudio ≥ 2.2 puede fallar
`from torchaudio.backend.common import AudioMetaData`. Si ocurre, instalar
`torch<2.2 torchaudio<2.2` en el entorno virtual.

## Uso rápido

El script incluido ejecuta la cadena completa:

```bash
python .claude/skills/limpiar-audio-voz/scripts/limpiar_voz.py entrada.mp4 salida.wav
```

Opciones útiles:

| Opción | Por defecto | Para qué |
|---|---|---|
| `--hum` | `60` | Frecuencia de red (usar `50` en Europa) |
| `--sin-wpe` | off | Saltar WPE si la sala ya está seca |
| `--wpe-taps` | `10` | Más taps = más reverberación eliminada (y más lento) |
| `--nr` | `0.5` | Fuerza de noisereduce (0–1). Mantener ≤ 0.6 |
| `--lufs` | `-16` | Sonoridad integrada objetivo |
| `--guardar-intermedios` | off | Guarda cada etapa para comparar |

## Pasos manuales (referencia)

### 1–2. Decodificar y quitar zumbido

```bash
ffmpeg -y -i entrada.ext -vn -ac 1 -ar 48000 \
  -af "highpass=f=60:poles=2,\
equalizer=f=60:t=q:w=30:g=-30,equalizer=f=120:t=q:w=30:g=-25,\
equalizer=f=180:t=q:w=30:g=-20,equalizer=f=240:t=q:w=30:g=-15,\
equalizer=f=300:t=q:w=30:g=-12" \
  -c:a pcm_f32le 01_sin_zumbido.wav
```

### 3. WPE (48 kHz)

Ventana STFT de 2048 / salto 512 (≈43 ms / 10.7 ms a 48 kHz):

```python
import numpy as np, soundfile as sf
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft

y, sr = sf.read("01_sin_zumbido.wav", dtype="float32")
assert sr == 48000
Y = stft(y[np.newaxis, :], size=2048, shift=512).transpose(2, 0, 1)  # (F, C, T)
Z = wpe(Y, taps=10, delay=3, iterations=3, statistics_mode="full")
z = istft(Z.transpose(1, 2, 0), size=2048, shift=512)[0, : len(y)]
sf.write("02_wpe.wav", z, 48000, subtype="FLOAT")
```

### 4. DeepFilterNet3

```python
from df.enhance import enhance, init_df, load_audio, save_audio
model, df_state, _ = init_df(model_base_dir="DeepFilterNet3")
assert df_state.sr() == 48000
audio, _ = load_audio("02_wpe.wav", sr=df_state.sr())
save_audio("03_dfn3.wav", enhance(model, df_state, audio), df_state.sr())
```

### 5. noisereduce suave con los frames más silenciosos

Dividir en frames de 50 ms, calcular RMS, tomar el 10 % más silencioso
(mínimo ~0.5 s en total) y usarlo como `y_noise`:

```python
import noisereduce as nr
frame = int(0.05 * sr)
n = len(x) // frame
rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
k = max(int(n * 0.10), int(0.5 / 0.05))
idx = np.sort(np.argsort(rms)[:k])
ruido = np.concatenate([x[i * frame:(i + 1) * frame] for i in idx])
limpio = nr.reduce_noise(y=x, sr=48000, y_noise=ruido,
                         stationary=True, prop_decrease=0.5)
```

### 6. Normalizar a -16 LUFS (dos pasadas)

```bash
# Pasada 1: medir
ffmpeg -i 04_nr.wav -af loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json -f null -
# Pasada 2: aplicar con los valores medidos (input_i, input_tp, input_lra, input_thresh, target_offset)
ffmpeg -y -i 04_nr.wav -af "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=...:measured_TP=...:measured_LRA=...:measured_thresh=...:offset=...:linear=true" \
  -ar 48000 -c:a pcm_s24le salida.wav
```

## Verificación

```bash
ffprobe -v error -show_entries stream=sample_rate,channels -of default=nw=1 salida.wav   # sample_rate=48000
ffmpeg -i salida.wav -af ebur128=peak=true -f null - 2>&1 | grep -E "I:|Peak:"         # I ≈ -16 LUFS, pico ≤ -1.5 dBTP
```

Escucha siempre el resultado: si la voz suena metálica o "bajo el agua",
bajar `--nr` (p. ej. 0.3) o desactivar WPE con `--sin-wpe`.

## Problemas frecuentes

- **Zumbido persiste** → comprobar si es de 50 Hz (`--hum 50`) o con más
  armónicos; añadir notches extra.
- **Voz con "bombeo" o cortes** → noisereduce demasiado fuerte; bajar `--nr`.
- **Eco sigue presente** → subir `--wpe-taps` a 15–20.
- **Salida no está a 48 kHz** → falta `-ar 48000` tras `loudnorm`.
