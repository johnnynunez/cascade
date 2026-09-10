# Cómo abrir la demo

La entrega usa **DGX Spark/Linux + Isaac Sim 6.1.0.0 + Newton +
Cosmos3-Edge + OpenClaw**. El launcher abre el chat al terminar el proof;
`--no-open` evita abrirlo. La UI de cámaras es independiente y opcional.
El Mac sirve para desarrollo; no certifica la ejecución GPU de la entrega.

---

## 0. Un clic (lo que se enseña a un visitante)

### DGX Spark

Después de publicar estos cambios en el ref de distribución:

```bash
curl -fsSL https://raw.githubusercontent.com/johnnynunez/cascade/main/scripts/bootstrap.sh | bash -s -- --accept-eula
```

Para N máquinas, fija un commit probado tanto en la URL como en `--ref`.
Esta sesión no publica los cambios ni certifica Spark: aún no hay equipo
accesible. Desde este checkout se usa:

```bash
bash scripts/install.sh --dir "$PWD" --profile spark --accept-eula
# Sólo preparar paquetes/modelos antes del evento, sin arrancar servicios:
bash scripts/install.sh --dir "$PWD" --profile spark --accept-eula --prepare-only
```

El instalador pide aceptación explícita de la licencia y no cambia drivers.
Aísla CASCADE, Isaac y Cosmos; instala OpenClaw 2026.9.3 localmente y usa
el perfil `cascade-demo`, no la configuración personal. Si Cosmos falla,
no lo sustituye por OpenAI. Detalles y aceptación pendiente:
[SPARK_DELIVERY.md](SPARK_DELIVERY.md).

### Desarrollo en el Mac o un rig ya instalado

```bash
git clone https://github.com/johnnynunez/cascade && cd cascade
./run.sh mujoco --brain keep     # usa la autenticación ya configurada en OpenClaw
./run.sh mujoco --cameras mujoco_scene_two    # la escena de dos cubos (demo de memoria)
./run.sh check isaac            # sólo informa qué falta; no instala ni descarga nada
./run.sh down                   # para todo lo que arrancó (sidecars + servidores MCP de sesiones viejas)
```

La primera vez crea el venv, instala los extras del modo, baja los meshes
del robot, instala/actualiza la CLI de OpenClaw, arranca los sidecars
(occupancy :5557, GraspGen-X stub :5556), registra las herramientas del
robot y **prueba el stack antes de decir READY**: construye el runtime con
el mismo entorno que recibe el servidor, lista las 41 tools, saca una
respuesta trivial del cerebro y ejecuta pick y reset en una misma sesión.
El receipt `proof.json` debe vincular modelo, sesión y proceso MCP con
el resultado físico y la restauración del objeto manipulado. Un puerto
abierto, una frase del modelo o un trace de otra sesión no bastan. Con
`--no-robot-turn` sólo muestra STARTED / UNVERIFIED, nunca READY. Las
siguientes ejecuciones reutilizan los paquetes y assets descargados.

El banner READY dice lo que hay de verdad: backend de occupancy, planner de
agarre (stub o modelo), número de tools, URL del chat, y si la ventana de
MuJoCo está abierta o por qué no (pantalla bloqueada → se abre en el primer
movimiento tras despertarla).

Luego abre `http://127.0.0.1:18789/` y habla, o sin navegador:

```bash
openclaw agent exec "what do you see?"
openclaw agent exec "pick and place the red object"
openclaw agent exec "did it actually move?"
# una sesión con memoria entre turnos (lo que hace el dashboard):
openclaw agent -m "put both cubes in the drop zone, one at a time; call task_memory before each action" --session-id demo-1
openclaw agent -m "how many did you move and how do you know?" --session-id demo-1
openclaw agent -m "reset the scene" --session-id demo-1
```

Cada llamada a herramienta que hace el host queda en
`runs/mcp_<pid>/server.log` (OpenClaw sólo enseña un contador de fallos).

### La demo de memoria (dos cubos)

Un pick-and-place es markoviano: nadie ve trabajar la memoria. Con dos
props, tras el primer pick la vista actual sola no dice si se movió uno o
ninguno. Di en el chat:

> Put both cubes in the drop zone, one at a time. Before each action call
> task_memory to see what you already did, and when both are done tell me
> how many cubes you moved and how you know.

El cerebro ve hasta K=4 frames de lo que ya hizo (estado inicial, la vista
tras cada acción, el veredicto de la física) junto a la vista actual, y su
respuesta cita esos veredictos. Medido: 2 `pick_and_place` confirmados por
física, 0 fallos, ~100 s.

Entre visitantes: **"reset the scene"** ("start over", "reinicia la
escena") — brazo a home, props al spawn, world model y memoria de tarea
limpios. En Isaac se exige read-back físico finito de cada prop dentro de
la tolerancia de spawn. El reflejo funciona sin LLM en la CLI de CASCADE;
en el chat OpenClaw sigue siendo el modelo del host quien elige la tool.

---

## 0-bis. Sin hardware, sin GPU, sin servidores, sin OpenClaw

```bash
uv venv && uv pip install -e '.[dev,kinematics]' && source .venv/bin/activate
python -m cascade.apps.demo --arm so101_mock --camera mock_small \
    --task "pick and place the red object"
```

Eso ejecuta la cascada completa sobre un SO-101 de 5 ejes simulado
cinemáticamente. Con `.[sim]` + `python scripts/fetch_robot_assets.py so101`
pasas a física real en MuJoCo (`--arm so101_mujoco --camera mujoco_scene`):
la cámara está RENDERIZADA desde el mismo mundo que pisa el brazo y la
postcondición lee la pose real del prop → `postcondition: confirmed
(channel: physics)`. Con `.[sim-warp]`, `--arm so101_mjwarp` es el MISMO
MJCF sobre MuJoCo Warp (GPU; en CPU ~650× más lento que el motor C — vale
para desarrollar esa ruta, no para enseñar la demo).

---

## 1. El rig de referencia (reBot / Isaac Sim + cerebro local)

Requisitos que arrancan una vez, cada uno en su terminal:

```bash
export ISAACSIM_PATH=~/Projects/isaac/IsaacSim/_build/linux-x86_64/release
./run.sh isaac                          # arranca el bridge (:8611) y todo lo demás
# o a mano:
$ISAACSIM_PATH/python.sh scripts/isaac_bridge.py     # Newton por defecto; --engine physx

scripts/serve_qwen_llamacpp.sh          # Qwen3.6 -> :8080   (uno de los dos)
scripts/serve_cosmos_vllm.sh            # Cosmos3-Edge -> :8082
TORCH_CUDA_ARCH_LIST=12.0 scripts/serve_graspgenx.sh franka_panda 5556   # agarres 6-DoF aprendidos
```

Comprobar que todo está vivo (sin `ss`, que en macOS no existe):

```bash
python - <<'EOF'
import socket
for p in (8611, 8080, 8082, 5556, 5557, 18789):
    s = socket.socket(); s.settimeout(0.3)
    print(p, "up" if s.connect_ex(("127.0.0.1", p)) == 0 else "down"); s.close()
EOF
```

`./run.sh isaac --brain auto` usa el servidor local si responde y, si no,
la autenticación que OpenClaw ya tenga.

## 2. Una orden desde la terminal (sin chat host)

```bash
cd models && python -m cascade.apps.demo \
    --cameras isaac,isaac_side --arm isaac --llm local_qwen \
    --task "pick and place the pink cube in the box" --no-view
```

> `cd models` no es opcional con el detector YOLOE: busca `mobileclip_blt.ts`
> en el CWD. Con las cámaras mock/MuJoCo no hace falta.

Perfiles de cerebro: `--llm hermes` (Nous Portal, `NOUS_API_KEY`) |
`local_qwen` | `local_cosmos` | `local_cosmos_sglang` | `anthropic` |
`openai` | `mock` (comprobación de cableado, sin LLM). Por defecto `auto`:
Hermes, Anthropic u OpenAI según qué clave esté exportada; sin ninguna,
`mock`. El CLI corre las tres capas (reflejo → hábito → LLM); el chat host
sólo la suya.

## 3. Chat interactivo en la terminal

```bash
cd models && python -m cascade.apps.demo \
    --cameras isaac,isaac_side --arm isaac --llm local_qwen --interactive
```

Entiende español e inglés: `describe the scene` / `¿qué ves?` · `coge el
cubo rosa y ponlo en la caja` · `abre las cámaras` (devuelve una URL) ·
`lanza la banana` · `reinicia la escena`.

---

## Las cámaras: cerradas hasta que las pidas

Perception **nunca para** (el rig bombea frames y el world model sigue
caliente), pero no se bindea ningún puerto hasta que alguien quiere mirar.

| skill | qué hace |
|---|---|
| `analyze_scene` | "¿qué ves?" **sin abrir nada**: detecciones + calidad de profundidad + descripción |
| `open_live_view` / `close_live_view` / `live_view_status` | dashboard en el navegador, y liberar el puerto |
| `probe_point(u,v)` | **cursor**: qué hay en ese píxel, distancia, si es alcanzable |
| `annotated_view` | marcas numeradas + rejilla de 5 cm + región alcanzable |
| `task_memory` (sólo MCP) | los K frames de lo que ya hizo en esta tarea, con veredicto |
| `world_state` (sólo MCP) | objetos, qué sostiene, último camino de despacho |

El dashboard se **auto-cierra a los 15 min** sin nadie mirando. Vistas:
**rgb** (detector + HUD), **depth** (colormap + min/mediana/máx + % válido),
**agent** (marcas, rejilla, banda IK). Más panel analyze, world model,
narración y un chat que maneja el mismo brazo.

`stream.mode` en `configs/demo.yaml`; `CASCADE_STREAM` manda encima:
`lazy` (por defecto) · `eager` (bindea al arrancar; `booth.yaml` lo fija) ·
`off` (`CASCADE_STREAM=0`, kill switch). `CASCADE_BOOTH=1` = modo feria.

Ventanas nativas: la de **MuJoCo** se abre con `CASCADE_MJ_VIEW=1` (el
launcher lo pone en modo sim; en macOS bajo `mjpython`). Si la pantalla
está bloqueada al arrancar, se salta con motivo en el log y se abre en el
primer movimiento tras despertarla — abrirla a ciegas segfaulteaba el
servidor. La ventana cv2 de cámaras no se abre desde el servidor en macOS
(Cocoa exige el hilo principal); usa el dashboard.

---

## Parar el brazo

- Dashboard: botón rojo **stop**
- MCP/OpenClaw: herramienta `emergency_stop` (fuera de banda: no espera a
  que acabe el movimiento); `reset_stop` lo desactiva (`CASCADE_HIDE_TOOLS=reset_stop`
  lo hace sólo-staff)
- Terminal: `Ctrl+C` (soft-stop; otra vez para salir)
- Cancelar el turno en el host a mitad de movimiento también congela el
  brazo — y deja el e-stop puesto: di `reset_stop` antes del siguiente.

---

## Si algo falla

```bash
python -m pytest tests/ -q                    # 737 passed / 2 deselected / 0 skipped, ~4 min
python scripts/learn_from_runs.py --report    # qué falló últimamente y por qué
./run.sh check mujoco                         # preflight de sólo lectura
tail -f runs/mcp_*/server.log                 # cada tools/call del host, con resultado
```

- **El turno del visitante falla con "Connection closed"** → el servidor MCP
  murió a mitad de llamada. Mira `~/Library/Logs/DiagnosticReports/mjpython-*.ips`
  (macOS) y `/tmp/openclaw/openclaw-<fecha>.log`. Causa conocida y cerrada:
  abrir la ventana de MuJoCo con la pantalla dormida.
- **El pick se cancela a los 60 s y luego todo falla con e-stop** → el host
  tiene `requestTimeoutMs` en el default (60 s); el launcher registra 300 s.
  Reregistra con `./run.sh <modo>` y `reset_stop`.
- **Cada turno tarda 10 s más de lo debido** → una entrada MCP muerta en
  `~/.openclaw/openclaw.json`; el launcher las poda al arrancar.
- **Puerto 8090 ocupado** → un run antiguo sigue vivo. `CASCADE_STREAM_PORT=8097`.
- **`no frame yet`** → el rig aún calienta; espera 2-3 s.
- **YOLOE no encuentra pesos** → no arrancaste desde `models/`.
- **El agente no llama a ninguna herramienta con Cosmos** → el perfil debe ser
  `type: cosmos3`, no `openai_compat` (emite tool calls en XML).
- **La lista de tools está obsoleta tras cambiar el registro** → `openclaw
  gateway restart`; el launcher lo hace por ti.
