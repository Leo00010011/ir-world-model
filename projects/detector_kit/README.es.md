# Detector kit: entrena y usa nuestro detector DINO-DETR con dos comandos

> 🇬🇧 English version: [README.md](README.md). Los comandos, opciones y mensajes de error están en
> inglés porque es lo que verás en la terminal.

Esta carpeta te permite usar el detector de objetos del proyecto `ir-world-model` sin tener que
aprender detrex, detectron2, Hydra ni el resto del repositorio. Dos scripts lo hacen todo:

```bash
bash projects/detector_kit/train.sh     --modality thermal            # entrenar un detector en FLIR
bash projects/detector_kit/inference.sh --head det_ir --images DIR    # detectar objetos en tus imágenes
```

**Qué es el detector.** Una cabeza DINO-DETR (`dino_vitdet` de
[detrex](https://github.com/IDEA-Research/detrex)) que no mira los píxeles directamente. Primero,
un codificador DINOv2-Reg ViT-B/14 **congelado** (+ una PCA a 1152 canales) convierte cada imagen
en features, y solo se entrena la cabeza. Predice 6 clases: `person, bike, car, motor, light, sign`
(persona, bicicleta, coche, moto, semáforo, señal). Hay dos detectores ya entrenados:

| nombre | entrenado con | AP en FLIR val | AP50 en FLIR val |
|---|---|---|---|
| `det_rgb` | FLIR ADAS v2, cámara RGB | 0.302 | 0.573 |
| `det_ir` | FLIR ADAS v2, cámara térmica | 0.330 | 0.626 |

**No necesitas los kernels de CUDA compilados.** Normalmente detrex necesita una operación de
CUDA compilada con `nvcc` (la *multi-scale deformable attention*). Si falta, los dos scripts lo
detectan y usan la versión en PyTorch puro de esa misma operación que trae detrex. Los resultados
son los mismos; solo es más lento y usa más memoria de GPU (ver
[Velocidad y memoria](#velocidad-y-memoria)). El log te dice cuál se está usando:
`deformable attention: cuda` (compilada) o `deformable attention: pytorch` (alternativa en PyTorch).

---

## 1. Configuración inicial (una sola vez)

### Qué necesitas
- **Linux x86_64** con `git`, `bash` y un compilador de C++ (`gcc`). La primera ejecución compila
  el código C++ de detectron2/detrex con `gcc`. **No** hace falta `nvcc` ni el CUDA toolkit.
- **Internet** en la primera ejecución (paquetes de Python y ~350 MB de pesos del codificador). En
  un clúster, haz la primera ejecución en el nodo de login.
- **Para entrenar:** una GPU NVIDIA (driver ≥ 525) y el dataset FLIR ADAS v2.
- **Para inferencia:** la GPU es opcional (funciona en CPU, pero lento). El dataset FLIR solo hace
  falta para `--split`.
- Unos 10 GB de disco para el entorno de Python (la mayor parte es PyTorch).

### Pasos
1. **Haz un fork y clónalo con los submódulos** (detrex y DINO-Foresight son submódulos de git):
   ```bash
   git clone --recurse-submodules <URL de tu fork> ir-world-model
   cd ir-world-model
   ```
   Si ya lo clonaste sin submódulos: `git submodule update --init --recursive`.

2. **Crea el archivo `.env`** en la raíz del repositorio. Le dice al código dónde está cada cosa:
   ```bash
   cp .env.example .env
   ```
   Después edita las cuatro rutas (cada una en su propia línea; sin comentarios después del valor):
   ```bash
   # carpeta que CONTIENE teledyne_flir_adas_v2/ (solo lectura)
   IRWM_DATA_ROOT=/ruta/a/datasets
   # cachés (puede estar en cualquier sitio)
   IRWM_CACHE_ROOT=/ruta/a/irwm/cache
   # pesos de modelos: aquí se descargan los del codificador y se guardan tus checkpoints
   IRWM_CKPT_ROOT=/ruta/a/irwm/checkpoints
   # logs, métricas y resultados de inferencia
   IRWM_OUTPUT_ROOT=/ruta/a/irwm/outputs
   ```
   En un clúster, ponlas en el almacenamiento de scratch/trabajo, no en tu directorio home.

3. **Consigue el dataset (solo para entrenar y para `--split`).** Descarga FLIR ADAS v2 (Teledyne
   FLIR, registro gratuito), descomprímelo y llama a la carpeta `teledyne_flir_adas_v2`, de forma
   que exista esta ruta: `$IRWM_DATA_ROOT/teledyne_flir_adas_v2/FLIR_ADAS_v2/images_thermal_train/coco.json`.

4. **Instala los detectores entrenados (necesarios para inferencia con `--head`).** Los pesos no
   están en git: Leonardo te los pasa en un zip, `irwm_frozen_detectors.zip` (≈ 170 MB). Dentro,
   los archivos ya están en la estructura de carpetas que espera el código:
   ```
   detection/frozen/det_rgb/head.pt     # pesos del detector RGB
   detection/frozen/det_rgb/meta.json   # su configuración, métricas en val y checksum SHA-256
   detection/frozen/det_ir/head.pt      # pesos del detector térmico
   detection/frozen/det_ir/meta.json
   ```
   Descomprímelo **dentro de la carpeta a la que apunta `IRWM_CKPT_ROOT`** (la que pusiste en
   `.env`, paso 2):
   ```bash
   source slurm/env.sh                      # carga IRWM_CKPT_ROOT desde .env en esta terminal
   mkdir -p "$IRWM_CKPT_ROOT"
   unzip irwm_frozen_detectors.zip -d "$IRWM_CKPT_ROOT"
   ls "$IRWM_CKPT_ROOT"/detection/frozen/*  # debe mostrar det_ir/ y det_rgb/, cada una con head.pt y meta.json
   ```
   No cambies el nombre de los archivos ni de las carpetas, y no edites `meta.json`. Cada vez que
   se carga un detector, el código comprueba que `head.pt` coincide con el checksum de `meta.json`
   y se detiene con un error `SHA-256` si no coincide (por ejemplo, tras una descarga incompleta;
   basta con descomprimir de nuevo). Tampoco los subas a git: los archivos `*.pt` están ignorados a
   propósito. Para entrenar con `train.sh` no hacen falta.

5. **Ejecuta cualquiera de los dos scripts.** La primera ejecución hace el resto sola:
   - instala `uv` (el gestor de paquetes de Python de este repositorio) en `~/.local/bin` si hace falta,
   - crea el entorno de Python 3.11 en `.venv/` (5–15 min; aquí se compilan detectron2 y detrex),
   - descarga el codificador congelado (pesos de DINOv2-Reg + PCA) en `$IRWM_CKPT_ROOT`.

   Una forma rápida de hacerlo y de comprobar que todo funciona:
   ```bash
   bash projects/detector_kit/train.sh --modality thermal --quick --batch-size 2
   ```

---

## 2. Inferencia: `inference.sh`

```bash
# tus propias imágenes (una carpeta, se busca también en subcarpetas), con el detector térmico
bash projects/detector_kit/inference.sh --head det_ir --images /ruta/a/frames

# imágenes RGB, solo con CPU
bash projects/detector_kit/inference.sh --head det_rgb --images /ruta/a/rgb --device cpu

# evaluar un detector en un split de FLIR (mAP COCO contra las etiquetas)
bash projects/detector_kit/inference.sh --head det_ir --split val

# un detector que entrenaste tú con train.sh
bash projects/detector_kit/inference.sh --ckpt /ruta/a/best-e034.ckpt --images /ruta/a/frames

# entre modalidades: el detector RGB sobre imágenes térmicas
bash projects/detector_kit/inference.sh --head det_rgb --images /ruta/a/termicas --modality thermal
```

| opción | qué hace | por defecto |
|---|---|---|
| `--head det_rgb\|det_ir` | un detector entrenado de `$IRWM_CKPT_ROOT/detection/frozen/` | hace falta `--head` o `--ckpt` |
| `--ckpt RUTA` | un checkpoint creado por `train.sh` | |
| `--images DIR` | todas las `.jpg .jpeg .png .bmp .tif .tiff` dentro de `DIR` | hace falta `--images` o `--split` |
| `--split val\|test\|train` | un split de FLIR; además calcula el mAP | |
| `--modality rgb\|thermal` | cómo se leen las imágenes (térmica → escala de grises, copiada a 3 canales) | la modalidad del detector |
| `--device auto\|cuda\|cpu` | dónde se ejecuta | `auto` (GPU si hay) |
| `--batch-size N` | imágenes por pasada; bájalo si te quedas sin memoria de GPU | 8 |
| `--workers N` | procesos de carga de datos (solo `--split`) | min(8, núcleos de CPU) |
| `--score-thr X` | guardar y dibujar cajas con score ≥ X (modo imágenes) | 0.3 |
| `--max-draw N` / `--no-draw` | cuántas imágenes guardar con las cajas dibujadas | 100 |
| `--msda auto\|cuda\|pytorch` | forzar la operación compilada o la alternativa en PyTorch | `auto` |
| `--out DIR` | carpeta de salida | `$IRWM_OUTPUT_ROOT/detector_kit/<fecha>/<hora>` |

**Salidas** (la última línea del log muestra la carpeta):
- `detections.json`
  - `--images`: una entrada por imagen, con las cajas de score ≥ `--score-thr`:
    ```json
    {"file": "seq1/frame_0001.jpg",
     "detections": [{"bbox": [315.5, 231.0, 38.3, 32.8], "score": 0.967, "label": "car", "category_id": 3}]}
    ```
    `bbox` es `[x, y, ancho, alto]` en píxeles de la imagen original, con `(x, y)` = esquina
    superior izquierda. `category_id` usa los ids de FLIR (person 1, bike 2, car 3, motor 4,
    light 10, sign 12).
  - `--split`: la lista estándar de resultados COCO, las 300 cajas por imagen, sin umbral (lo que
    necesita el mAP).
- `metrics.json` (solo `--split`): AP, AP50, AP75, AP por tamaño y por clase.
- `vis/*.jpg`: las imágenes con las cajas dibujadas.
- `inference.log`: el log completo.

Cómo se preparan las imágenes: cada imagen se recorta por el centro a una proporción 5:4 y se
redimensiona a 448×560, exactamente igual que en el entrenamiento. Las cajas se devuelven en las
coordenadas de tu imagen original. El detector térmico espera imágenes térmicas de 8 bits como las
de FLIR (claro = caliente). Con imágenes muy distintas a las de FLIR (otras cámaras, paletas de
color o puntos de vista) los resultados serán peores.

---

## 3. Entrenamiento: `train.sh`

```bash
# primero comprueba la instalación: 1 época corta, tarda unos minutos
bash projects/detector_kit/train.sh --modality thermal --quick --batch-size 2

# la receta completa (50 épocas) en térmico; ponle nombre al run para poder reanudarlo
bash projects/detector_kit/train.sh --modality thermal --batch-size 8 --name ir_bs8

# ¿se paró (error, límite de tiempo)? el mismo comando con el mismo --name lo reanuda
bash projects/detector_kit/train.sh --modality thermal --batch-size 8 --name ir_bs8

# experimento más corto, registrado en Weights & Biases
bash projects/detector_kit/train.sh --modality rgb --epochs 20 --val-every 2 --wandb
```

| opción | qué hace | por defecto |
|---|---|---|
| `--modality rgb\|thermal` | con qué cámara de FLIR entrenar | obligatoria |
| `--epochs N` | número de épocas; la tasa de aprendizaje baja ×0.1 al 80 % de ellas | 50 |
| `--batch-size N` | imágenes por paso y por GPU (ver la tabla de memoria más abajo) | 16 |
| `--workers N` | procesos de carga de datos | min(8, núcleos de CPU) |
| `--gpus N` | GPUs de esta máquina (más de 1 usa DDP) | 1 |
| `--lr X` | tasa de aprendizaje | 1e-4 |
| `--val-every N` | calcular el mAP en FLIR val cada N épocas | 5 |
| `--name NOMBRE` | nombre del run = carpeta de checkpoints; el mismo nombre reanuda | fecha y hora |
| `--msda auto\|cuda\|pytorch` | forzar la operación compilada o la alternativa en PyTorch | `auto` |
| `--wandb` | registrar también en W&B (ejecuta `uv run wandb login` una vez) | desactivado |
| `--quick` | prueba de instalación: 1 época, 20 lotes de entrenamiento y 10 de validación | desactivado |
| `--out DIR` | carpeta para logs y métricas | `$IRWM_OUTPUT_ROOT/detector_kit/<fecha>/<hora>` |

Todo lo que pongas después de `--` se pasa al script de Python como override de Hydra (las claves
están en [`conf/train.yaml`](conf/train.yaml)), por ejemplo `-- data.flip_p=0.0 optim.weight_decay=5e-4`.

**Salidas:**
- Checkpoints en `$IRWM_CKPT_ROOT/detector_kit/<nombre>/`: `best-eNNN.ckpt` (la época con mejor
  AP en val) y `last.ckpt` (para reanudar). Usa `best-eNNN.ckpt` con `inference.sh --ckpt`.
- En la carpeta de salida: `val_metrics/epoch_NNN.json` (mAP de cada validación),
  `csv/version_0/metrics.csv` (pérdidas y métricas, listas para graficar), `train_summary.json`
  (mejor AP y ruta del mejor checkpoint), `train.log`.

**Cuál es la receta.** La misma que la de nuestros `det_rgb`/`det_ir`: el schedule DINO de 50
épocas de detrex, AdamW (lr 1e-4, weight decay 1e-4), batch 16, precisión mixta (fp16), gradient
clipping 0.1, volteos horizontales aleatorios y codificador congelado. Con un batch distinto de 16
la tasa de aprendizaje no se reescala, así que espera resultados parecidos, pero no idénticos, a
los de la tabla del principio.

### Velocidad y memoria

Medido en una NVIDIA L4 (24 GB). "cuda" es la operación compilada; "pytorch" es la alternativa.

| | operación compilada | alternativa en PyTorch |
|---|---|---|
| velocidad de entrenamiento (solo la cabeza, fp16) | ≈ 13 img/s | ≈ 7 img/s (1.7–1.9× más lento) |
| velocidad de inferencia (solo la cabeza, fp16) | ≈ 50 img/s | ≈ 20 img/s |
| un entrenamiento completo de 50 épocas | ≈ 14 h (≈ 16 min por época) | aproximadamente 1.5–2× más |

La memoria de GPU durante el entrenamiento es sobre todo la de la cabeza. Picos aproximados (cabeza, fp16):

| batch size | operación compilada | alternativa en PyTorch | cabe en |
|---|---|---|---|
| 2 | 2.5 GiB | 4.1 GiB | GPUs de 6–8 GB |
| 4 | 4.7 GiB | 7.8 GiB | GPUs de 10–12 GB |
| 8 | 9.1 GiB | 15.2 GiB | GPUs de 16–24 GB |
| 16 | ≈ 18 GiB | ≈ 30 GiB | 24 GB con la operación compilada; 40 GB o más con la alternativa |

Suma 1–2 GB para el codificador y para CUDA. La inferencia necesita mucho menos: con batch 1 funciona
en una GPU de 4 GB con la alternativa en PyTorch.

### Opcional: compilar los kernels

Solo merece la pena para entrenar más rápido. Necesitas el CUDA toolkit 12.x (`nvcc` en el `PATH`,
por ejemplo `module load cuda/12.4` en un clúster) y `gcc` ≤ 12. Después recompila:
```bash
bash slurm/setup_env.sh
```
Detecta `nvcc` y compila las operaciones de CUDA para V100, T4, A100, RTX 30xx/40xx, L4 y más nuevas
(`TORCH_CUDA_ARCH_LIST` en `slurm/setup_env.sh`; añade la compute capability de tu GPU si es más
antigua). La siguiente ejecución debería mostrar `deformable attention: cuda`.

---

## 4. Ejecutar en un clúster SLURM

Los scripts se ejecutan en la máquina donde se lanzan, así que dentro de un job funcionan tal cual:
```bash
#!/bin/bash
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=24:00:00
#SBATCH --output=train-%j.out
cd /ruta/a/ir-world-model
bash projects/detector_kit/train.sh --modality thermal --batch-size 8 --name ir_bs8
```
Crea el entorno y descarga los pesos antes, una vez, en el nodo de login (paso 5 de la
configuración), porque los nodos de cómputo suelen no tener internet. Como el run tiene un `--name`
fijo, si vuelves a enviar el mismo job después de un time-out, se reanuda.

---

## 5. Solución de problemas

| mensaje | qué hacer |
|---|---|
| `set IRWM_... in .env` | paso 2 de la configuración |
| `frozen detector not found` | paso 4 de la configuración (descomprimir dentro de `$IRWM_CKPT_ROOT`) |
| `SHA-256 ... != meta.json head_sha256` | `head.pt` está dañado o es de otra versión: vuelve a descomprimir el zip |
| `CUDA out of memory` | baja `--batch-size` (ver la tabla de memoria) |
| `no CUDA GPU visible` (entrenamiento) | entrenar necesita una GPU NVIDIA; revisa `nvidia-smi` |
| `encoder download failed` | la primera ejecución necesita internet; hazla una vez en una máquina o nodo de login que lo tenga |
| `FileNotFoundError ... coco.json` | `IRWM_DATA_ROOT` debe ser la carpeta que contiene `teledyne_flir_adas_v2/` |
| falla la instalación en detectron2/detrex | necesitas `gcc`/`g++` (`sudo apt install build-essential`, o `module load gcc` en un clúster) |
| W&B pide iniciar sesión | no uses `--wandb`, o ejecuta `uv run wandb login` una vez |
| cualquier otra cosa | el log completo está en la carpeta de salida (`train.log` / `inference.log`); pon `HYDRA_FULL_ERROR=1` delante del comando para ver el traceback completo de Python |

---

## 6. Dónde está el código (si quieres cambiar algo)

| archivo | qué hace |
|---|---|
| `projects/detector_kit/train.sh`, `inference.sh` | leen las opciones y llaman a los scripts de Python de abajo |
| `projects/detector_kit/_common.sh` | configuración en la primera ejecución: `.env`, entorno de Python, descarga del codificador |
| `projects/detector_kit/train.py`, `conf/train.yaml` | punto de entrada del entrenamiento y su configuración |
| `projects/detector_kit/inference.py`, `conf/inference.yaml` | punto de entrada de la inferencia y su configuración |
| `lib/irwm/detection/detector.py` | el detector (`DINODetector`): el DINO de detrex con una entrada de features |
| `lib/irwm/detection/lightning.py` | bucle de entrenamiento (`DetectorModule`, PyTorch Lightning) |
| `lib/irwm/detection/detrex_adapter.py` | carga de detrex y el cambio entre operación compilada y alternativa en PyTorch (`select_msda`) |
| `lib/irwm/features/encoder.py`, `preprocess.py` | el codificador congelado DINOv2-Reg + PCA y el preprocesado de imágenes |
| `lib/irwm/data/detection.py`, `flir.py` | lectura de FLIR y el dataset de entrenamiento |

Para entrenar con un dataset distinto de FLIR, empieza por `lib/irwm/data/detection.py`: el modelo
solo necesita lotes de `images [B, 3, 448, 560]` (normalizadas con ImageNet) y, por imagen,
`boxes [N, 4]` (x1, y1, x2, y2 en esos píxeles de 448×560) con `labels [N]` en `0..5`.
