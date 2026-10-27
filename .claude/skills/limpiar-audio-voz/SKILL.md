---
name: limpiar-audio-voz
description: Limpia grabaciones y aísla la voz trabajando siempre a 48 kHz (nunca baja a 16 kHz). Cadena con ffmpeg, eliminación de zumbido de 60 Hz, WPE multicanal para el eco/reverberación, DeepFilterNet3 con post-filtro para aislar la voz, noisereduce con perfil de los frames más silenciosos, mezcla en seco contra el sonido robótico, EQ/compresión de podcast, expander por energía para atenuar voces de fondo lejanas y normalización a -16 LUFS. Úsala cuando pidan limpiar un audio, quitar ruido, eco o zumbido, quitar voces de fondo, aislar o mejorar la voz de una grabación, entrevista, podcast o vídeo.
---

# Limpiar audio y aislar la voz

Cadena de procesado para dejar una voz limpia, nítida "de podcast", sin
zumbido, con poco eco, con las voces de fondo lejanas atenuadas y a volumen
estándar de podcast/redes (-16 LUFS).

## Regla innegociable: 48 kHz en todo momento

- Todo el procesado se hace a **48 000 Hz**. **Nunca** remuestrear a 16 kHz
  (ni a 22.05/24/44.1 kHz) en ningún paso intermedio ni en la salida. Si la
  entrada viene a 44.1 kHz se **sube** a 48 kHz al decodificar.
- DeepFilterNet3 trabaja nativamente a 48 kHz: no hace falta (ni se debe)
  bajar la frecuencia para él.
- `loudnorm` de ffmpeg sobremuestrea internamente a 192 kHz; por eso la salida
  **siempre** lleva `-ar 48000` explícito.
- Formato intermedio: WAV float 32 (`pcm_f32le`) para no perder rango ni
  provocar clipping entre etapas.
- Al terminar, verifica con `ffprobe` que la salida es 48000 Hz.
- Única excepción: la medición opcional con DNSMOS (ver "Medir la calidad")
  analiza una **copia** a 16 kHz; el audio entregado nunca se toca.

## Orden de la cadena (y por qué)

1. **Decodificar con ffmpeg a 48 kHz float, conservando los canales.** Acepta
   cualquier entrada (mp3, m4a, mp4, mov, wav…). Si es estéreo se mantiene
   estéreo para el paso 3.
2. **Quitar zumbido de 60 Hz**: notch en 60 Hz y armónicos (120–300 Hz) +
   pasa-altos suave a 60 Hz. Para red de 50 Hz (Europa) usar `--hum 50`.
3. **WPE (nara_wpe) multicanal para el eco.** Con estéreo, WPE quita bastante
   más reverberación que sobre la mezcla mono (10 taps, delay 3, 5
   iteraciones). Va antes que los denoisers porque es un predictor lineal y
   necesita la señal sin procesar de forma no lineal. Después se mezcla a mono.
4. **DeepFilterNet3 con post-filtro** para aislar la voz: suprime más el
   ruido y las voces de fondo que van por debajo de la voz principal.
5. **noisereduce** con perfil de los **frames más silenciosos** (no de los
   primeros segundos): `stationary=True`, `prop_decrease=0.6`. No pasar de 0.6.
6. **Mezcla en seco**: sumar la señal tras WPE (sin red neuronal) a **-20 dB**.
   Devuelve armónicos que DFN3 y noisereduce se comen y quita el timbre
   "robótico" o "bajo el agua". Más alto (-18) = más natural pero más fondo.
7. **EQ/compresión de podcast** (ffmpeg): pasa-altos 80 Hz, -2 dB en 280 Hz
   (barro), +2.5 dB en 3.5 kHz (presencia/nitidez), +1.5 dB shelf en 9 kHz
   (aire), compresor 2.5:1 suave y de-esser. Va **antes** del expander: si
   el compresor va después, vuelve a subir el fondo.
8. **Expander por energía** para las voces lejanas (ver abajo).
9. **Limitador + `loudnorm`** a -16 LUFS en dos pasadas (I=-16, TP=-1.5,
   LRA=11) y salida a 48 kHz. Sin el limitador previo el pico verdadero
   impide llegar al objetivo y se queda ~1 dB corto.

### Expander para voces de fondo

Las voces de fondo suenan más bajas y lejanas que las principales; el
expander atenúa lo que queda muy por debajo de la voz principal:

- Detección en la banda de voz (150–4000 Hz), envolvente RMS de 10 ms.
- **Umbral relativo al nivel local** de la voz principal (percentil 90 en una
  ventana de ~4 s), no un umbral fijo: si una de las voces principales habla
  más bajo o más lejos, un umbral global la hundiría.
- **Hold causal** (sólo después de la voz) + **release** para no cortar el
  final de las palabras, y **lookahead** de 15–20 ms para no comerse los
  ataques. Un hold simétrico deja la puerta abierta casi siempre en habla
  densa y el expander no hace nada.

| Perfil | Umbral bajo el nivel local | Ratio | Atenuación máx. | Hold | Release |
|---|---|---|---|---|---|
| `fuerte` (por defecto) | 5 dB | 6:1 | 35 dB | 50 ms | 120 ms |
| `suave` | 10 dB | 3:1 | 15 dB | 100 ms | 200 ms |

Límite físico: un expander sólo baja el fondo **en las pausas**; lo que suena
a la vez que la voz principal sólo lo puede reducir DeepFilterNet3.

### Lo que NO conviene hacer

- **Supresión espectral de reverberación tardía** (modelo exponencial
  Lebart/Habets, resta de la cola) después de DFN3: quita algo más de eco,
  pero deja la voz claramente robótica (DNSMOS SIG bajó de 2.86 a 2.35 en
  pruebas). WPE multicanal + mezcla en seco da mejor equilibrio.
- WPE con muchos taps (20) o delay 2: más eco quitado pero peor timbre.
- Encadenar dos pasadas de DeepFilterNet3: más fondo quitado, voz más
  artificial.

## Instalación

```bash
# ffmpeg (sistema)
sudo apt-get install -y ffmpeg        # macOS: brew install ffmpeg

# Python
pip install numpy scipy soundfile nara_wpe noisereduce deepfilternet torch torchaudio
# opcional, para medir la calidad
pip install onnxruntime
```

Notas DeepFilterNet:
- Con torchaudio ≥ 2.2 falla `from torchaudio.backend.common import
  AudioMetaData`; el script lo resuelve creando ese módulo vacío antes de
  importar `df`. Con torchaudio ≥ 2.9 tampoco existen `torchaudio.info`/`load`,
  por eso el script lee y escribe con soundfile y llama a `enhance` directamente.
- Si la descarga del modelo falla (403 a `github.com`), bajarlo de
  `https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3.zip`
  y descomprimirlo en `~/.cache/DeepFilterNet/` (debe quedar
  `~/.cache/DeepFilterNet/DeepFilterNet3/config.ini`).

## Uso rápido

```bash
python .claude/skills/limpiar-audio-voz/scripts/limpiar_voz.py entrada.wav salida.wav
```

Opciones útiles:

| Opción | Por defecto | Para qué |
|---|---|---|
| `--hum` | `60` | Frecuencia de red (usar `50` en Europa) |
| `--sin-wpe` | off | Saltar WPE si la sala ya está seca |
| `--wpe-taps` | `10` | Más taps = más eco quitado (y timbre peor) |
| `--wpe-mono` | off | WPE sobre la mezcla mono aunque la entrada sea estéreo |
| `--sin-postfiltro` | off | DeepFilterNet3 sin post-filtro (menos supresión) |
| `--nr` | `0.6` | Fuerza de noisereduce (0–1). No pasar de 0.6 |
| `--seco` | `-20` | dB de señal tras WPE mezclada de vuelta; `-99` = nada |
| `--sin-podcast` | off | Sin EQ/compresión/de-esser |
| `--expander` | `fuerte` | `fuerte`, `suave` o `no` |
| `--lufs` | `-16` | Sonoridad integrada objetivo |
| `--guardar-intermedios` | off | Guarda cada etapa para comparar |

Para entregar varias versiones a comparar, lo habitual es una con
`--expander suave` y otra con el perfil por defecto.

## Medir la calidad (DNSMOS)

Sin poder escuchar, DNSMOS P.835 da una estimación objetiva: **SIG**
(naturalidad de la voz; baja cuando suena robótica), **BAK** (ruido y voces
de fondo) y **OVRL** (global), de 1 a 5. Comparar siempre a nivel igualado.

```bash
curl -sSL -o sig_bak_ovr.onnx \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
python .claude/skills/limpiar-audio-voz/scripts/medir_dnsmos.py --modelo sig_bak_ovr.onnx a.wav b.wav
```

Referencia (entrevista con dos voces y voces de fondo lejanas, 44.1 kHz
estéreo): original SIG 1.75 / BAK 1.33; cadena por defecto SIG 2.77 /
BAK 3.16 / OVRL 2.21 antes de normalizar.

## Verificación

```bash
ffprobe -v error -show_entries stream=sample_rate,channels -of default=nw=1 salida.wav   # sample_rate=48000
ffmpeg -i salida.wav -af ebur128=peak=true -f null - 2>&1 | grep -E "I:|Peak:"         # I ≈ -16 LUFS, pico ≤ -1.5 dBTP
```

Escucha siempre el resultado.

## Problemas frecuentes

- **Voz robótica o "bajo el agua"** → subir la mezcla en seco (`--seco -18`),
  bajar `--nr` (0.4) o usar `--sin-postfiltro`.
- **Aún se oyen voces de fondo en las pausas** → `--expander fuerte` (por
  defecto); si ya está, bajar `--seco` a -24 (la mezcla en seco también trae
  algo de fondo).
- **Se cortan los finales de palabra** → `--expander suave`.
- **Una voz principal queda atenuada** → está muy por debajo de la otra;
  usar `--expander suave` o `--expander no`.
- **Eco sigue presente** → subir `--wpe-taps` a 15 (peor timbre por encima).
- **Zumbido persiste** → comprobar si es de 50 Hz (`--hum 50`).
- **Salida no está a 48 kHz** → falta `-ar 48000` tras `loudnorm`.
