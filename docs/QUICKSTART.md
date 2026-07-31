# Cómo abrir la demo

Tres formas de hablar con el brazo, de menos a más montaje. **Las cámaras y el
dashboard están cerrados por defecto** en todas: el chat es la interfaz, la UI
se abre cuando la pides.

---

## 0. Requisitos que arrancan una vez

```bash
cd ~/Projects/demo/wrc_demo
PY=~/Projects/demo/.demo/bin/python          # el venv compartido
```

**Isaac Sim (el rig simulado)** — proceso largo, déjalo en su terminal:
```bash
export ISAACSIM_PATH=~/Projects/isaac/IsaacSim/_build/linux-x86_64/release
$ISAACSIM_PATH/python.sh scripts/isaac_bridge.py --engine physx
# sirve en :8611
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
cd models && PYTHONPATH=../src $PY -m wrc_demo.apps.demo \
    --cameras isaac,isaac_side --arm isaac --llm local_qwen \
    --task "pick and place the pink cube in the box" --no-view
```

> `cd models` no es opcional: YOLOE busca `mobileclip_blt.ts` en el CWD.

Perfiles de cerebro: `--llm local_qwen` | `--llm local_cosmos` | `--llm mock`
(mock = comprobación de cableado, sin LLM).

## 2. Chat interactivo en la terminal

```bash
cd models && PYTHONPATH=../src $PY -m wrc_demo.apps.demo \
    --cameras isaac,isaac_side --arm isaac --llm local_qwen --interactive
```

Prueba a decirle (entiende español e inglés):
- `describe the scene` / `¿qué ves?`
- `coge el cubo rosa y ponlo en la caja`
- `abre las cámaras` → te devuelve una URL
- `lanza la banana`

## 3. OpenClaw / Hermes (la interfaz real)

```bash
scripts/openclaw_demo.sh                    # sim + cerebro cosmos
scripts/openclaw_demo.sh --brain qwen       # sim + Qwen
scripts/openclaw_demo.sh --arm rebot_rs --cameras l515   # brazo REAL (¡ojo!)
```

Abre `http://127.0.0.1:18789/` y habla. Registra el servidor MCP, apunta el
agente al cerebro local y levanta el gateway.

Probar un turno sin navegador:
```bash
openclaw agent --agent main -m "describe the scene"
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

`stream.mode` en `configs/demo.yaml`, y `WRC_STREAM` manda por encima:

| modo | comportamiento |
|---|---|
| `lazy` *(por defecto)* | no bindea hasta que lo pides; auto-cierra al inactivar |
| `eager` | bindea al arrancar — fijado en `booth.yaml` para día de feria |
| `off` | nunca bindea (`WRC_STREAM=0` = kill switch duro) |

```bash
WRC_STREAM=eager ...    # dashboard desde el segundo cero
WRC_STREAM=0 ...        # nunca, ni aunque lo pidan
WRC_BOOTH=1 ...         # modo feria completo (incluye eager)
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

- **Puerto 8090 ocupado** → un run antiguo sigue vivo. `WRC_STREAM_PORT=8097`.
- **`no frame yet`** → el rig aún calienta; espera 2-3 s.
- **YOLOE no encuentra pesos** → no arrancaste desde `models/`.
- **El agente no llama a ninguna herramienta con Cosmos** → el perfil debe ser
  `type: cosmos3`, no `openai_compat` (emite tool calls en XML).
