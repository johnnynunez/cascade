# Screw manipulation research — 2026-10-01

**Reusable assets and physical threading examples exist. The strongest first demonstration is a Factory nut on a fixed bolt, with measured thread advancement; seating and tightening torque require a separate validation stage.** A screwdriver task additionally needs a compatible bit/head, axial force control and slip detection. This investigation did not install or run simulators, download asset packages, or modify Cascade.

**SimReady.com: verified catalogue entries, unverified physical geometry**

SimReady.com is operated by Lightwheel Limited. Its public semantic search returned these published Lightwheel-YCB entries for `screw`, `nut` and `screwdriver`. The responses identify USD content delivered in ZIP files. Neither collider quality nor functional threads were inspected. [Asset library](https://simready.com/asset-library), [public search API, POST endpoint](https://simready.com/api/open/asset/usd_search).

| Exact asset / ID | Declared ZIP size | Possible use, subject to inspection |
|---|---:|---|
| `Lightwheel_plastic_bolt` / 1338 | 916,240 bytes | Bolt; thread profile and fit remain unverified |
| `Lightwheel_plastic_nut` / 1292 | 249,883 bytes | Nut; pairing with the bolt is not yet demonstrated |
| `Lightwheel_phillips_screwdriver` / 1411 | 3,837,558 bytes | Phillips tool; tip/head compatibility remains unverified |
| `Lightwheel_flat_screwdriver` / 1439 | 5,778,491 bytes | Flat tool; a matching screw is still needed |
| `Lightwheel_power_drill` / 1429 | 3,019,350 bytes | Tool body; the listing does not establish an actuated chuck |

All five records declare price zero and **CC BY-NC 4.0**. That licence requires attribution and restricts commercial use; the catalogue's `isOpenSource=true` flag does not remove those terms. The site's [terms, section 6](https://simready.com/protocol) distinguish individual asset licences and commercial creator EULAs. Its download UI requests login; no login or ZIP download was attempted. [Official CC BY-NC 4.0 terms](https://creativecommons.org/licenses/by-nc/4.0/).

Search is not exhaustive and includes irrelevant matches. No complete, physically validated screw/insert/bit/fixture assembly was established. Individual detail endpoint attempts returned API code 500, so no product permalink is invented. Use the catalogue names and IDs above. The [catalogue evidence](evidence/screw-research/catalogue.json) retains the selected fields and raw-response hashes; temporary signed download links are deliberately omitted.

**Primary examples with actual thread-contact support**

| Source | Available implementation | Scope limitation |
|---|---|---|
| [Isaac Lab Factory / FORGE](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html#contact-rich-manipulation) | `Isaac-Factory-NutThread-Direct-v0` and `Isaac-Forge-NutThread-Direct-v0`, Franka threading a nut onto the first two bolt threads. FORGE adds end-effector force observations, excessive-force penalties and dynamics variation. | The published task is not final seating, preload or tightening-torque acceptance. A Franka policy does not directly control Cascade's ReBot. |
| [Factory M16 configuration](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py#L327) | `Factory/factory_nut_m16.usd`, `Factory/factory_bolt_m16.usd`; declared pitch 0.002 m and explicit physical configuration. | These USD files were not downloaded; per-asset licence, cooking and regional availability remain to be checked. Isaac Lab's code licence does not establish the licence of every external USD. |
| [Newton nut/bolt SDF example](https://github.com/newton-physics/newton/blob/main/newton/examples/contacts/example_nut_bolt_sdf.py) | `m20_loose`, meshes `factory_bolt_m20_loose.obj` and `factory_nut_m20_loose_subdiv_3x.obj`, SDF collisions; MuJoCo/XPBD solver options. Example code declares Apache-2.0. | A contact demonstration, not a robot tightening controller. Its checks test nut motion/descent, not seating torque. Not executed here or verified against Cascade's installed Newton version. |
| [Original Factory meshes](https://github.com/isaac-sim/IsaacGymEnvs/tree/main/assets/factory/mesh/factory_nut_bolt) | Nut/bolt meshes used by the Newton example, plus other assembly resources. | IsaacGymEnvs is archived; reusing its geometry does not require adopting its obsolete runtime. |

The original Factory nut/bolt meshes have an explicit licence chain: [Factory acknowledgments](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/assets/licenses/factory-acknowledgments.txt) place them under the repository's [BSD-3-Clause licence](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/LICENSE.txt). Preserve the licence, attribution and asset hashes. This is more concrete than assuming all USD downloads have permissive redistribution terms.

**What Cascade already does**

Reviewed current `src/cascade/skills/runtime.py:3224–3370` and `tests/test_turn_screw.py`: `turn_screw` understands tighten/loosen, localizes a head, approaches, then opens, winds back the wrist, closes and rotates. `turns_applied` accumulates commanded last-joint travel. If contact-pose IK fails, it can continue at the hover pose. It does not measure fastener rotation, axial advancement, tightening torque, preload, engagement or bit slip. Tests use camera/arm doubles, not physical threaded fasteners. Its own docstring calls it geometric/gestural. [Pinned implementation](https://github.com/johnnynunez/cascade/blob/0e2387070c2b784ba864981f5c291a1b1e4d117a/src/cascade/skills/runtime.py#L3224), [tests](https://github.com/johnnynunez/cascade/blob/0e2387070c2b784ba864981f5c291a1b1e4d117a/tests/test_turn_screw.py).

Intent parsing, profiles, localization, IK, limits and evidence infrastructure are reusable. A dedicated assembly controller and outcome verifier are missing. The current return value must not be presented as proof that a screw was tightened.

**Proposed physical validation — not implemented**

1. **Thread geometry:** fix the bolt in a fixture, select a known mating nut, and validate masses, inertia, friction and colliders that preserve the thread profile. PhysX documents SDF collision support for dynamic triangle meshes on the GPU pipeline. A convex envelope filling the thread grooves cannot reproduce threading. Validate SDF resolution, offsets and convergence against the real pitch rather than copying another object's parameters. [PhysX geometry](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Geometry.html#triangle-meshes).
2. **Control and engagement:** align axes, limit axial force, rotate slowly, and measure relative fastener angle and axial travel. During engagement compare `delta_z` with signed `pitch * delta_theta / (2*pi)`. Record contact loss, jams and fixture motion. Wrist angle is an actuator state, not a substitute for fastener angle.
3. **Tightening:** add seating surfaces and a suitable stiffness/preload model; measure reaction torque about the fastener axis and axial force. Distinguish thread friction from a hard stop. Fix acceptance tolerances and force/torque limits before the trial from the asset/material/controller contract. A stopped joint or stationary image alone does not prove tightening torque.
4. **Reduced model option:** a helical coupling can impose rotation/translation through constraints. PhysX documents revolute/prismatic and rack-and-pinion joints; simply selecting an assumed generic screw joint is not an implementation. Such a coupling may be useful but cannot validate contact-driven cross-threading. Also, standard joint `getForce()` excludes drive-generated forces: it is not automatically a motor-torque measurement. [PhysX joints, force reporting and drives](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Joints.html).
5. **Evidence:** start with one isolated world; record mesh/cooking hashes, engine version, steps, poses, forces/torques, state-bound camera packets and an independent result. Include negative tests for misalignment, blocking and bit loss. Compare timestep/resolution before claiming fidelity; validate x86 and ARM separately if both are targets.

A practical first deliverable is **a gripper rotating a Factory nut on a fixed bolt, with observed advancement**, without a new trained policy or a claim of final tightening. Follow with measured seating/torque. Add an actual screwdriver only when a compatible tool/head, mount/TCP and axial controller are defined. OVRTX can render the scene; it does not supply contact physics or the outcome checks above.
