# Cómo abrir la demo

Tres formas de hablar con el brazo, de menos a más montaje. **Las cámaras y el
dashboard están cerrados por defecto** en todas: el chat es la interfaz, la UI
se abre cuando la pides.

Este documento describe el **rig de referencia** (reBot/Isaac + cerebro local).
Si sólo quieres ver el framework funcionando, no necesitas nada de esto: mira
la sección 0-bis.

---

## 0-bis. Sin hardware, sin GPU, sin servidores

```bash
uv venv && uv pip install -e '.[dev,kinematics]' && source .venv/bin/activate
python -m cascade.apps.demo --arm so101_mock --camera mock_small \
    --task "pick and place the red object"
```

Eso ejecuta la cascada completa sobre un brazo SO-101 de 5 ejes simulado
cinemáticamente. Con `pip install -e '.[sim]'` y
`python scripts/fetch_robot_assets.py so101` puedes cambiar a física real en
MuJoCo (`--arm so101_mujoco`), que funciona igual en CPU.

---

## 0. Requisitos que arrancan una vez

```bash
cd <tu-checkout>/cascade
PY=.venv/bin/python                          # o el intérprete que uses
```

**Isaac Sim (el rig simulado)** — proceso largo, déjalo en su terminal:
```bash
export ISAACSIM_PATH=~/Projects/isaac/IsaacSim/_build/linux-x86_64/release
$ISAACSIM_PATH/python.sh scripts/isaac_bridge.py
# sirve en :8611 con Newton (por defecto). Usa --engine physx para el motor
# anterior; ver docs/NEWTON_ENGINE.md.
```

**Cerebro local** — uno de los dos:
```bash
scripts/serve_qwen_llamacpp.sh          # Qwen3-VL  -> :8080
scripts/serve_cosmos_vllm.sh            # Cosmos3-Edge -> :8082
```

**Agarres 6-DoF (opcional, mejora mucho)**:
```bash
TORCH_CUDA_ARCH_LIST=12.0 scripts/serve_graspgenx.sh franka_panda 5556
```

Comprobar que todo está vivo:
```bash
ss -ltnp | grep -E '8611|8080|8082|5556'
```

---

## 1. Una orden y ya (lo más rápido)

```bash
cd models && PYTHONPATH=../src $PY -m cascade.apps.demo \
    --cameras isaac,isaac_side --arm isaac --llm local_qwen \
    --task "pick and place the pink cube in the box" --no-view
```

> `cd models` no es opcional: YOLOE busca `mobileclip_blt.ts` en el CWD.

Perfiles de cerebro: `--llm hermes` (Nous Portal, necesita `NOUS_API_KEY`) |
`--llm local_qwen` | `--llm local_cosmos` | `--llm anthropic` | `--llm mock`
(mock = comprobación de cableado, sin LLM). Por defecto `--llm auto`: usa
Hermes, Anthropic u OpenAI según qué clave esté exportada, y si no hay ninguna
cae en `mock`.

## 2. Chat interactivo en la terminal

```bash
cd models && PYTHONPATH=../src $PY -m cascade.apps.demo \
    --cameras isaac,isaac_side --arm isaac --llm local_qwen --interactive
```

Prueba a decirle (entiende español e inglés):
- `describe the scene` / `¿qué ves?`
- `coge el cubo rosa y ponlo en la caja`
- `abre las cámaras` → te devuelve una URL
- `lanza la banana`

## 3. OpenClaw / Hermes — **la interfaz principal**

```bash
scripts/openclaw_demo.sh --brain qwen       # sim + Qwen3-VL
scripts/openclaw_demo.sh                    # sim + Cosmos3-Edge
scripts/openclaw_demo.sh --arm rebot_rs --cameras l515   # brazo REAL (¡ojo!)
```

El script registra el MCP, levanta el gateway, apunta el agente al cerebro
local, lo reinicia para que cargue las 35 herramientas del robot, y **prueba
que responden** antes de decirte que funciona.

Luego abre `http://127.0.0.1:18789/` y habla. O sin navegador:

```bash
openclaw agent --agent main -m "¿qué ves?"
openclaw agent --agent main --session-key "agent:main:demo-$(date +%s)" \
    -m "Coge el cubo rosa y ponlo en la caja."
```

### Cuatro trampas que cuestan una demo (todas encontradas en vivo)

1. **El gateway debe arrancar ANTES de `onboard`** — si no, aborta con
   `Gateway did not become reachable`.
2. **Y reiniciarse DESPUÉS de `mcp add`** — si no sirve una lista de
   herramientas obsoleta y el agente responde en prosa sin llamar a nada.
3. **`--cwd` debe ser `models/`, NO la raíz del repo.** YOLOE resuelve
   `mobileclip_blt.ts` relativo al CWD. Desde la raíz no ve el fichero de
   600 MB, intenta descargarlo (aun con `YOLO_OFFLINE=True`), escribe un
   fichero truncado de 9 MB en la raíz, y toda observación muere con
   `PytorchStreamReader failed reading zip archive`. El agente lo reporta como
   *"a runtime error loading a checkpoint"* y deja de percibir en silencio.
   Si te pasa: `rm mobileclip_blt.ts` de la raíz del repo.
4. **`contextWindow` debe coincidir con el `n_ctx` real del servidor.**
   `onboard` escribe 128000 pase lo que pase; el script ahora lo pregunta a
   `/props`. Si no, los turnos largos revientan con *context overflow*.

Una sesión larga acumula contexto: usa `--session-key` nueva para empezar
limpio.

## 4. Terminal directo (sin OpenClaw)

```bash
# una orden y ya   (ojo: cd models, YOLOE busca sus pesos en el CWD)
cd models && PYTHONPATH=../src $PY -m cascade.apps.demo \
    --cameras isaac,isaac_side --arm isaac --llm local_qwen \
    --task "pick and place the pink cube in the box" --no-view

# chat interactivo
... --interactive
```

---

## Las cámaras: cerradas hasta que las pidas

Perception **nunca para** (el rig bombea frames y el world model sigue
caliente), pero no se bindea ningún puerto hasta que alguien quiere mirar.

Pídeselo al agente en lenguaje natural — o llama a las skills:

| skill | qué hace |
|---|---|
| `analyze_scene` | "¿qué ves?" **sin abrir nada**: detecciones + calidad de profundidad + descripción |
| `open_live_view` | abre el dashboard y devuelve la URL |
| `close_live_view` | lo cierra y libera el puerto |
| `live_view_status` | ¿está abierto? ¿qué URL? ¿cuánto lleva inactivo? |
| `probe_point(u,v)` | **cursor**: qué hay en ese píxel, a qué distancia, si es alcanzable |
| `annotated_view` | marcas numeradas + rejilla métrica + banda IK |

El dashboard se **auto-cierra a los 15 min** sin nadie mirando.

### Lo que verás al abrirlo

Todas las cámaras juntas, y cada una conmutable entre tres vistas:

- **rgb** — detecciones + HUD (lo que ve el *detector*)
- **depth** — mapa de color + min/mediana/máx + % válido (lo que ve la *geometría*)
- **agent** — marcas numeradas, rejilla de 5 cm, banda IK (sobre lo que *razona el agente*)

Más: panel **analyze**, el world model, la narración del robot, y un **chat**
que maneja el mismo brazo.

### Modos del dashboard

`stream.mode` en `configs/demo.yaml`, y `CASCADE_STREAM` manda por encima:

| modo | comportamiento |
|---|---|
| `lazy` *(por defecto)* | no bindea hasta que lo pides; auto-cierra al inactivar |
| `eager` | bindea al arrancar — fijado en `booth.yaml` para día de feria |
| `off` | nunca bindea (`CASCADE_STREAM=0` = kill switch duro) |

```bash
CASCADE_STREAM=eager ...    # dashboard desde el segundo cero
CASCADE_STREAM=0 ...        # nunca, ni aunque lo pidan
CASCADE_BOOTH=1 ...         # modo feria completo (incluye eager)
```

---

## Parar el brazo

- Dashboard: botón rojo **stop**
- MCP/OpenClaw: herramienta `emergency_stop`
- Terminal: `Ctrl+C` (soft-stop; otra vez para salir)

---

## Si algo falla

```bash
$PY -m pytest tests/ -q                    # 224 tests, ~43 s
$PY scripts/learn_from_runs.py --report    # qué falló últimamente y por qué
```

- **Puerto 8090 ocupado** → un run antiguo sigue vivo. `CASCADE_STREAM_PORT=8097`.
- **`no frame yet`** → el rig aún calienta; espera 2-3 s.
- **YOLOE no encuentra pesos** → no arrancaste desde `models/`.
- **El agente no llama a ninguna herramienta con Cosmos** → el perfil debe ser
  `type: cosmos3`, no `openai_compat` (emite tool calls en XML).
