# MicroDuck en CASCADE — propuesta de integración Isaac Sim 6.1

**Estado: aprobado por el usuario en Telegram para implementación y PR; implementación y validación pendientes.**

## Objetivo y alcance

Añadir MicroDuck como robot móvil sin brazo, controlable desde el mismo MCP y runtime de CASCADE. La física debe ejecutarse dentro de Isaac Sim 6.1, tanto con PhysX como con Newton. El agente local decide acciones de alto nivel; una política oficial de locomoción controla las articulaciones. El usuario ha priorizado esta ruta frente a una integración limitada a MuJoCo o al robot físico.

El primer PR abarcará una integración completa en una arena plana propia: observación, equilibrio, marcha con velocidad y duración acotadas, giro con realimentación, parada prioritaria, verificación, trazas y cámaras. No se considerará terminado por cargar el USD, ejecutar ONNX o producir un vídeo. No incluirá SLAM/Nav2, navegación a objetos/píxeles, interacción con la cocina, agarres con el pico, patadas/volteretas/rollers, entrenamiento ni un driver físico. Esas capacidades no se anunciarán en las herramientas.

La cocina y sus defaults permanecerán intactos. No se moverá el pin de instalación Spark, no se detendrán servicios ajenos y no se usarán los endpoints o la ventana de medición de Codex. El PR se abrirá como integración lista sólo después de validar sus criterios; si un motor queda bloqueado, se informará del bloqueo sin publicar soporte ficticio.

## Fuentes y evidencia disponible

Revisión CASCADE de referencia: `c9147db84356de32d6a98c7a70a1aca3ecefe084`. Al terminar la investigación, `origin/main` era `5359405a607a14e9d30182309c40d30c3b95f80b`; el cambio intermedio es documentación/evidencia, no código de producto. Antes de implementar se volverá a obtener main y se creará un worktree aislado desde ese estado.

- Runtime oficial: https://github.com/pollen-robotics/microduck/tree/1fa84386f07884e27866411bc1ba166977bced95
- Modelos y referencia RL: https://github.com/pollen-robotics/microduck_rl/tree/8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec
- BAM fijado por el lockfile: https://github.com/Rhoban/bam/tree/62bd8ce12154340be97e06f7f41a0ca8f116d967
- Pesos oficiales: https://huggingface.co/pollen-robotics/microduck-policies/tree/d5a8b55033e157f1af2ed6bd5c1e435b770a8ee0
- Referencia comunitaria Isaac: https://github.com/osrbot/microduck-ros2-isaac/tree/a2e7f327344eda402467af83053f6e33ca6bcdb7

Se han leído fuentes y contrastado estáticamente modelos y pesos. El coordinador ha verificado los diez ONNX locales contra tamaño y SHA-256 LFS de la revisión HF fijada. **Al cierre de la auditoría inicial todavía no se habían ejecutado inferencia, conversión ni pruebas de locomoción; este párrafo describe esa fase, no la validación posterior.** La referencia comunitaria declara Isaac 6.0.1 y actuadores PD aproximados; no demuestra compatibilidad con 6.1, Newton ni fidelidad BAM.

## Contexto upstream aportado durante la implementación

El usuario indicó [IsaacLab #8161](https://github.com/isaac-sim/IsaacLab/pull/8161). Se inspeccionó el head `28aa1fca5843208ff9a67935695a4d5376e44d50`: BAM nativo Newton/MJWarp, con presupuesto de fricción aplicado en el solver y carga del paso anterior. Ésta es la primera referencia de reutilización para Newton; una ley numérica CPU propia sirve como contraste, no como sustituto físico del solver. El PR no soporta BAM en PhysX y sus pruebas de integración son kitless: la conexión con nuestro runtime Isaac Sim 6.1 sigue necesitando validación aislada.

La rama relacionada `AntoineRichard/IsaacLab@b74ae74c0927da538846076289031c0881431ff0` sí contiene assets y tareas MicroDuck, pero usa una API BAM distinta del PR actual. No mezclar revisiones ni heredar su etiqueta Apache para modelos: el README de su propio pin fuente `d424a0c899f6b33cbd3daeb279913134349c0b63` declara BY-SA-NC para los modelos 3D. El PR BAM-only excluye esos assets. No se han reproducido aquí los resultados de tests/benchmarks que declara el autor; faltan fixtures publicados para parte de sus pruebas.

Este contexto cambia la prioridad de reutilización, no el alcance aprobado ni los criterios de ambos motores. No se instalará ni actualizará nada en los entornos de Codex. La auditoría con fuentes está en el workspace de investigación `research/isaaclab-pr8161/CONTEXT_FOR_CASCADE.md`.

## Arquitectura recomendada

```text
OpenClaw / otro host MCP — agente local
                 |
     catálogo según capacidades del robot
                 |
       SkillRuntime.execute + trazas
                 |
       SafeBase + coordinador de parada
                 |
     puente MicroDuck privado en Isaac Sim
                 |
   observación -> ONNX -> modelo de actuador
                 |
       PhysX o Newton: raíz libre + contactos
                 |
  lectura física pasiva -> verificador -> resultado
```

### 1. Robot móvil real en la composición de CASCADE

Añadir `MobileBase`, `BaseState`, `MobileRig` y `SafeBase`, con perfiles bajo `configs/bases/`. El modo base-only no construirá un brazo mock, pinza, TCP, FK/IK de brazo ni memoria de grasp. Los arranques actuales sin selección móvil conservarán su comportamiento.

`BaseState` distinguirá pose mundial/odom, orientación completa, velocidades, estado articular, salud, tiempo físico, epoch y fuente. El origen móvil no se confundirá con el frame fijo de la mesa. En este PR las cámaras sirven para observación y evidencia; no se afirmará que existe un mapa persistente de navegación.

Separar los hooks de movimiento de brazo y base. Hoy `_MOTION_SKILLS` también implica selector `arm`, pinza, objetos agarrados y verificación de carry; añadir una marcha a ese set sin separar responsabilidades sería incorrecto. El catálogo y la ejecución rechazarán operaciones incompatibles incluso si alguien invoca manualmente una herramienta no anunciada.

### 2. Puente Isaac específico y opt-in

Crear una entrada de escena MicroDuck independiente, reutilizando helpers puros de arranque/identidad/atestación cuando sean compatibles. No importar ni parametrizar a ciegas `scripts/isaac_bridge.py`: actualmente contiene nombres y targets de seis articulaciones reBot, pinza, poses y resets propios.

Cada conexión comprobará robot, versión de protocolo, prim, motor/solver efectivos, hashes de asset/política/configuración, capacidades, timestep y epoch. Un endpoint reBot o una generación anterior no podrá aceptar una orden MicroDuck. Stop, estado/cámara y órdenes tendrán canales que no bloqueen entre sí.

El adaptador PhysX y el de Newton compartirán contrato externo, no supuestas equivalencias internas. Se comprobarán contactos, esfuerzo aplicado, masas/inercia, límites, resets, temporización y dispositivo reales por motor. La atestación Newton CUDA existente se reutilizará; no se deshabilitará para simular compatibilidad.

### 3. Política local ligada al tiempo físico

**Opción recomendada:** ONNX dentro del host Isaac, desacoplado del LLM y de la cámara. Física nominal de referencia a 200 Hz y política a 50 Hz, con observaciones de un mismo paso completo. El render lento no consumirá ticks de política sobre estado repetido. Los plazos de seguridad de pared y de simulación permanecerán separados.

El checkpoint inicial candidato es `velstand.onnx`, slot oficial para caminar y mantener comando cero; `alpha_walking`/`alpha_stand` serán referencias diagnósticas, no fallbacks silenciosos. La combinación exacta checkpoint/modelo/actuador debe pasar admisión y baseline antes de declararse perfil soportado. Si no la supera, no se sustituirá ni reentrenará sin documentar y revisar el cambio de alcance.

Contrato auditado: `obs float32[1,61] -> actions float32[1,14]`. Las observaciones contienen gyro y gravedad en el tronco, q−HOME, dq, acción anterior cruda y trece comandos. La normalización ya está en ONNX. Las acciones son offsets de objetivos angulares, no torques. El mapeo será por nombre: la boca es el servo físico adicional, no una acción locomotora; articulaciones pasivas tampoco son acciones.

El primer perfil admitirá el formato feed-forward realmente disponible. Redes recurrentes, dimensiones, joints o artefactos no admitidos fallarán antes de actuar en vez de recibir soporte nominal sin pruebas.

#### Contrato temporal de cancelación y trazas

- Stop invalida la **intención** inmediatamente. Una inferencia pendiente vuelve a comprobar identidad/epoch, los clocks del estado físico completo capturado, generación y comando antes de seleccionar su target. Una salida invalidada no cambia la acción RAW anterior. Se permite un único recálculo en el mismo slot con el mismo estado y la intención vigente; invalidaciones repetidas provocan fault/contención sin otro solve.
- La selección RAW bajo el lock corto de permisos es el punto de compromiso. No hay ONNX, subida de targets ni trabajo GPU bajo ese lock. Un stop posterior no deshace el paso ya en vuelo; entre slots de política se retiene el target previamente comprometido hasta la siguiente actualización a50Hz, preservando el delay físico admitido. No se borra BAM ni se cambia la cadencia para simular quietud inmediata.
- Hay un único propietario de física y un único `tick()` en vuelo. Comprobar clocks detecta avances ajenos, pero no constituye exclusión mutua universal contra escritores arbitrarios fuera de ese contrato.
- `max_wall_s` impide despachar otra inferencia, comprometer/subir otro target, preparar otro control o iniciar otro solve una vez vencido su presupuesto. Se comprueba después de llamadas lentas y dentro de la barrera precommit. Una llamada nativa ya iniciada puede terminar después; el presupuesto no puede interrumpirla. Un timeout externo del grupo de procesos sigue siendo obligatorio.
- Cada intento conserva el input61D real y su disposición. `attempt` distingue descarte y recálculo aunque compartan `observation_step`; no se deduplican por paso. `committed`/`policy_commits` indican selección RAW en memoria, no éxito de subida ni torque aplicado. `first_step_after_commit` sólo existe tras un solve completo. Un fallo de subida puede conservar `committed=true` y `status=failed`, sin ese primer paso. Si una señal interrumpe la copia de historia o su contabilidad antes de registrar su retorno, se publica `status=interrupted`, `committed=null` y `commit_outcome=unknown_due_to_interruption`: no se afirma que la historia permaneció intacta ni se continúa ese episodio.
- SIGINT/SIGTERM del CLI nativo son solicitudes de terminar el episodio, no ACKs de reposo. El handler sólo escribe estado escalar y provoca unwind; no usa `Event.set`, locks, RPC ni logging. La construcción pasiva y el teardown difieren la primera señal hasta publicar/cerrar sus recursos. La ejecución no se difiere: al entregarse la excepción en Python, se libera la sección crítica, se invalida permiso y se contiene/cierra sin otro dispatch. El recibo preservado antes de cerrar el SDK indica interrupción y el código `128+signal`. Una llamada nativa ya en vuelo puede acabar antes de que Python entregue la señal; no se promete preempción GPU ni quietud física. El timeout externo sigue siendo obligatorio.
- El CLI nativo protege explícitamente su registro de señales dentro de su propio proceso: el `SimulationApp` instalado intenta sustituir SIGINT durante su constructor por un callback que descarga el SDK y sale con código0. La protección opt-in conserva el handler de CASCADE y reinstala el dispatcher Python frente a cambios nativos de disposición, reenvía registros de señales no poseídas y restaura función/handlers al salir. No se edita el SDK ni se cambia el comportamiento por defecto de demo/MCP. La propiedad de señales en Kit real y el cierre del proceso requieren validación propia, aparte de la prueba CPU.
- `policy_target_generation` identifica el último target seleccionado, no la procedencia del esfuerzo retrasado. El retardo/estado de BAM debe reconstruirse o instrumentarse separadamente para atribuir torque. Un ACK sigue siendo físicamente `unverified`; sólo observación independiente posterior puede certificar reposo.

Estos límites son un contrato de implementación y prueba, no aceptación de marcha, parada física ni sim2real.

### 4. Actuadores y conversión: puertas de validación tempranas

Preservar modelo oficial, raíz libre, geometría de contacto, masa/inercia, ejes, rangos y procedencia en una conversión reproducible. No copiar automáticamente las desactivaciones de colisión del reBot ni las del ejemplo comunitario. No cambiar parámetros calibrados para disimular una conversión incorrecta.

La política se entrenó con BAM XL330/M6; un PD implícito es una aproximación diferente. Primero comparar observaciones y respuesta del actuador con una referencia ejecutable: escalón, rampa, reversión, carga, saturación y reposo. Se probará qué contratos pueden representarse realmente en PhysX y Newton antes de desarrollar la demostración completa.

Existe una discrepancia de fuente que debe resolverse explícitamente: `infer_policy.py` desactiva el límite de corriente, mientras el constructor BAM fijado lleva 1.75 A y la construcción de entrenamiento leída no lo elimina. Eso no demuestra cómo se entrenaron los pesos históricos. La baseline declarará y medirá su configuración efectiva; no se elegirá una interpretación silenciosamente.

Las variantes backlash exigen observación encoder = motor + holgura, mientras el helper CPU leído no reproduce por sí solo ese contrato. No entran automáticamente en el primer perfil. Ninguna aproximación PD, soporte cinemático, teletransporte o posición escrita durante el episodio contará como prueba de locomoción física o de fidelidad BAM.

### 5. Herramientas, seguridad y resultados

Superficie propuesta:

- `list_bases` y `get_base_state`: identidad, capacidades y estado; no activan actuadores.
- `walk_velocity(vx, vy, wz, duration_s, base=...)`: comando en frame del cuerpo, duración finita, límites de perfil y resultado medido.
- `turn(angle_rad, base=...)`: cierre sobre yaw medido, plazo finito y verificación; no integración de comandos presentada como giro real.
- `stop_navigation`: cancelación prioritaria hacia twist cero manteniendo el controlador de equilibrio validado.
- `emergency_stop`: invalida órdenes actuales/en cola y bloquea nuevas órdenes; el reset sólo permite órdenes nuevas, nunca reanuda las anteriores.
- Cámara, memoria de tarea y consulta de resultados, sin ofrecer habilidades de manipulación.

El stop actual de MCP sólo llama a `runtime.arm`; se reemplazará por coordinación según capacidades, conservando stop pendiente durante startup, cancel-before-dispatch y el canal lector prioritario. EOF, desconexión, lease vencido, clock obsoleto, estado no finito, inferencia fallida y cambio de epoch invalidarán el comando. El watchdog vive junto al controlador, no depende de que el LLM responda.

Para un bípedo, parada no significa torque-off ni congelar articulaciones. Si el controlador de equilibrio o los sensores dejan de ser fiables, se declarará fault y la contención del simulador se identificará como tal. Una simulación pausada o un soporte de arranque no demostrarán que el robot permanece de pie. No habrá auto-reset ni recuperación de caída que borre el fallo.

No se atribuirá evitación general de obstáculos a la occupancy de brazos. La arena es controlada y acotada. Cualquier comprobación de geofence basada en truth se identificará como seguridad de simulación, no percepción/autonomía transferible a hardware.

El verificador leerá física pasivamente por un canal separado: pose, velocidades, pasos avanzando, identidad/epoch y contactos. Sólo evidencia suficiente podrá producir `confirmed`; evidencia contraria dará `refuted`; falta de evidencia, `unverified`. Un ACK no será éxito y `unverified` no contará como aceptación.

## Criterios de aceptación del PR

### Software

- Arranque MicroDuck sin dependencias de brazo; defaults de cocina invariantes.
- Mapeo/observaciones/acciones contrastados con referencia oficial, sin normalización doble ni boca/pasivos incluidos como acciones.
- Catálogo MCP y CLI por capacidades y rechazo en ejecución.
- Stop/cancel/startup/EOF/timeout/reset/epoch probados, sin reactivación de comandos viejos.
- Teardown de workers, cámaras y sockets sin procesos/hilos huérfanos.
- Verificador con controles positivos, negativos y ausencia/falsedad de feedback.
- Suite completa y CI existentes, incluidas instalación mínima y plataformas actuales, sin degradar sus garantías.

### Física y ruta del agente

Campañas separadas PhysX y Newton, con condiciones comparables y archivos versionados:

1. Modelo importado, contactos de suelo y estado finito; ningún soporte invisible ni error de importación sin resolver.
2. Equilibrio, avance/retroceso, giro a ambos lados y parada, desde varios arranques limpios y después de reset. Movimiento lateral sólo si el perfil admitido lo soporta.
3. Medición de error de movimiento, deriva, inclinación/altura, velocidad residual tras parar, saturación y caídas. Umbrales fijados antes de las campañas a partir del dominio y baseline admitidos, nunca relajados para conseguir verde.
4. Cadencia física/política comprobada bajo render lento y pausa; comandos vencidos no sobreviven a la reanudación.
5. Cancelación/desconexión durante marcha y fallos de estado/política como negativos obligatorios; el resultado debe conservar el fallo.
6. Ruta real del usuario: host MCP -> runtime CASCADE -> SafeBase -> política -> física -> verificador -> trazas/memoria. Un script directo de joints no sustituye esta prueba.
7. Vídeo/capturas del mismo episodio que los JSONL/recibos; cada recibo identifica revisión, asset, política, motor, configuración, clocks y hardware. Resultados x86 no se llamarán resultados Spark.

Si cambia infraestructura compartida de Isaac o las rutas de ejecución/parada de la cocina, repetir sus dos casos y resets en ambos motores. No editar recibos previos ni aflojar `demo_proof.py`.

## Licencias y empaquetado

El código upstream y la model card de pesos tienen avisos Apache-2.0 separados. Los **modelos 3D** declaran literalmente `Creative Commons BY-SA-NC`, sin versión: https://github.com/pollen-robotics/microduck_rl/blob/8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec/README.md#L268-L271.

El PR no venderá esos assets como MIT/Apache ni incluirá meshes/USD derivados en el paquete CASCADE. Proveerá admisión y conversión desde una descarga/localización explícita, con origen, hashes y avisos preservados. Mantener los archivos fuera de git no elimina las condiciones de uso de los modelos. No se habilitarán por defecto en el presenter ni se afirmará permiso para uso comercial/eventos; eso requiere aclarar los términos con upstream/titular.

## Alternativas descartadas para este primer PR

- **Sólo robotd + MuJoCo:** útil como referencia, pero no satisface el target Isaac pedido.
- **robotd + body server Isaac:** seam real y viable para una fase posterior de compatibilidad, pero introduce un lazo de pared y un protocolo sin step/epoch lockstep; no simplifica la primera validación sim-only. El body server oficial actual usa PD, no es prueba de fidelidad BAM.
- **USD comunitario + PD sin admitir contratos:** acelera una preview visual, pero no prueba el controlador/actuador ni Newton y no será el criterio de terminado.
- **MicroDuck como ArmBase o herramienta exec genérica:** contradice capacidades, seguridad y verificación de CASCADE.

## Secuencia tras la aprobación

1. Worktree desde main actualizado; fijar alcance/archivos y recursos privados antes de lanzar nada.
2. Admisión de asset/peso y experimentos que puedan refutar la viabilidad de actuador/temporización en ambos motores.
3. Contratos móviles y composición base-only con pruebas primero.
4. Puente/política, seguridad, herramientas y verificador.
5. Campañas reales, vídeo, regresiones, revisión independiente, documentación y CI.
6. Commit/push y PR verificado. No merge automático.

Este documento define el diseño; no sustituye las pruebas ni autoriza a anunciar soporte que todavía no existe.
