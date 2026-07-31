# Synthetic-Augmented RGB-D → 3D Object Localization

Análisis del pipeline propuesto (imagen del 2026-07-31) frente a lo que este
repo ya tiene, con **el problema medido en el rig**, no estimado.

---

## El problema que ataca — medido, no supuesto

`scripts/eval_detector.py`, 12 frames, escena **completamente estática**,
ground truth de física Isaac (`TruthPoseReader` → PhysX, independiente de
percepción):

```
GROUND TRUTH (3 objetos)
   pink_cube   [0.17,  0.15, 0.04]
   green_cube  [0.30,  0.16, 0.04]
   bin         [0.18, -0.17, 0.03]

count por frame : [3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
estabilidad     : FLICKER
count modal     : WRONG  (4, deberian ser 3)

precision       : 76.6%
recall          : 100.0%
phantom rate    : 23.4%   (11 de 47 detecciones)
localizacion    : media 2.4 cm, peor 3.9 cm
```

Tres lecturas, y conviene separarlas:

1. **Recall 100%.** Nunca se pierde un objeto real. El detector no es ciego.
2. **Precision 76.6% — un fantasma persistente.** Ve 4 objetos donde hay 3,
   frame tras frame. No es ruido aleatorio: es un falso positivo *estable*,
   que entra en el world model y sobrevive. Ya se coló en la UI — es el 4º
   badge que salió en el `annotated_view` de la primera sesión.
3. **Flicker.** El primer frame ve 3, los demás 4. La misma escena da
   respuestas distintas, así que "¿cuántos cubos hay?" no tiene respuesta
   estable.

Y un dato que exculpa al bloque C: **localización media 2.4 cm**. El
levantamiento 2D→3D funciona bien. El problema está *aguas arriba*, en qué
máscaras produce el detector, no en cómo se convierten a 3D.

Un detector open-vocabulary genérico (YOLOE + prompts de texto) nunca vio
*esta* mesa, *estos* cubos, *esta* iluminación. Está haciendo zero-shot sobre
un dominio para el que no fue entrenado, y ese 23.4% es el precio.

**Ese es exactamente el hueco que llena el pipeline.**

---

## Qué propone, bloque a bloque

| bloque | qué hace | estado en este repo |
|---|---|---|
| **A** RGB-D bootstrapping | capturas eye-in-hand → etiquetado asistido por SAM → fine-tune → re-etiquetar → promover a train/val | ❌ no existe |
| **B** síntesis de vistas | levantar objetos a nubes de puntos, re-renderizar con pitch/yaw, componer escenas nuevas etiquetadas | ❌ no existe |
| **C** YOLO-Seg + 2D→3D lifting | máscara + profundidad + intrínsecos → centroide en base frame | ✅ **ya implementado** |
| **D** centroide para robótica | el centroide alimenta el agarre | ✅ **ya implementado** |

**El bloque C ya está entero** en `perception/grounding.py`:
`mask_to_points_cam()` retroproyecta píxeles de máscara con banda de
profundidad inter-cuantil (justo lo que el diagrama llama *lift objects to
point clouds*), y `oriented_bbox()` da centro + ejes por PCA. `probe.py`
añade `deproject()` con lectura de profundidad por mediana.

O sea: **el pipeline no propone reemplazar nada de lo que tienes. Propone
alimentar lo que ya funciona con un detector que no alucine.**

---

## El bucle de auto-etiquetado (bloque A) es lo valioso

La idea que hace esto barato es que **el propio robot genera las etiquetas**:

```
capturas eye-in-hand → SAM etiqueta (solo RGB) → fine-tune bootstrap
      ↑                                                    ↓
      └──────── revisar ← generar etiquetas nuevas ────────┘
```

Cada vuelta el modelo etiqueta mejor, así que la vuelta siguiente necesita
menos revisión humana. Es Voyager/ASPIRE aplicado a percepción en vez de a
skills — y encaja exactamente con el bucle exterior que ya montamos:
`scripts/learn_from_runs.py` + el cron nocturno.

**Ya tienes 80 keyframes acumulados** de las sesiones previas, gratis, con
pose de brazo y beliefs asociados. Ese es el arranque del bloque A sin
capturar nada nuevo.

Y algo que el diagrama no puede saber pero este repo sí: **tenemos ground
truth de física**. `TruthPoseReader` da la posición real de cada prop. Eso
convierte el "SELECTED SUBSET / review" manual del bloque A en un filtro
automático: una etiqueta cuyo centroide 3D no cae a <6 cm de un prop real es
basura y se descarta sola. En sim, el bucle A se cierra **sin humano**.

---

## Dónde discrepo del diagrama

**1. El bloque B (composición sintética) es el de peor relación coste/valor
aquí, y lo pondría el último.**

El pipeline lo justifica para conseguir diversidad de puntos de vista. Pero
este rig tiene **Isaac Sim**: puedo mover la cámara, cambiar iluminación,
randomizar poses de props y sacar *renders físicamente correctos con
etiquetas perfectas* — sin recortar-y-pegar nubes de puntos, que produce
composiciones con iluminación inconsistente y bordes de recorte que el modelo
aprende como atajo. Domain randomization nativa gana a composición 2.5D
cuando ya tienes el simulador.

El bloque B tiene sentido cuando tu dominio es **solo real** y no tienes
gemelo digital. Para el D435i/L515 sobre el brazo físico, sí. Para la demo en
sim, Isaac lo hace mejor.

**2. "Get centroid for robotic tasks" (D) es donde el diagrama es más débil.**

Un centroide es suficiente para *pointing*, no para *grasping*. Ya aprendimos
esto en este repo por las malas: el centroide de una nube de puntos parcial
está sesgado hacia la cara visible, y GraspGenX existe precisamente porque un
OBB no basta para elegir orientación de pinza. El pipeline no debería
sustituir a GraspGenX; debería **darle mejores máscaras**.

**3. Falta el bucle de verificación.** El diagrama es abierto: entrena,
despliega, fin. No dice cómo sabes que el modelo nuevo es mejor. Con
`TruthPoseReader` eso es medible: precision/recall contra física, antes y
después. Sin esa métrica, "promote to train set" es fe.

---

## Lo que haría, en orden

**1. Métrica primero — ✅ HECHO.** `scripts/eval_detector.py` puntúa
precision/recall/flicker/localización contra física. Baseline registrado
arriba. Sin esto, "promote to train set" es fe.

```bash
cd models && PYTHONPATH=../src python ../scripts/eval_detector.py \
    --frames 12 --json /tmp/det_baseline.json
```

**2. Auto-etiquetado en sim, sin humano (1-2 días).** Randomizar props en
Isaac, capturar RGB-D + máscaras + pose real, y **filtrar cada etiqueta
contra la verdad física**. Esto es el bloque A con el paso de revisión
automatizado — el sim regala lo que en el mundo real cuesta un humano.

**3. Fine-tune YOLOE sobre ese conjunto y volver a medir con (1).**
Objetivo concreto y falsable: **precision de 76.6% a >95%**, y `count_stable`
= true en 12 frames. Si no se mueve, el enfoque no valía y lo sabremos.

**4. Enganchar al cron nocturno que ya existe.** `learn_from_runs.py` ya
corre a las 03:00. Que también acumule keyframes etiquetados y dispare un
fine-tune cuando haya suficientes nuevos. Ahí se cierra el bucle A del
diagrama, sin intervención.

**5. Bloque B solo para el brazo real**, donde no hay gemelo digital.

---

## Veredicto

La idea es **buena y ataca un problema real y medido de este repo** (49% de
fantasmas). El bloque C que propone ya lo tienes construido y probado, lo cual
es una señal de que el diseño es correcto — coincide con lo que ya hacía falta.

El valor está en **A**, no en B. Y en este rig, A sale mucho más barato de lo
que el diagrama asume, porque la física da las etiquetas gratis.
