# cProfile of the shared owner stepping loop (arm P2, PR #225 build, 12 robots, 1000 completed steps)

total profiled time 75.3 s (cProfile inflates call-heavy Python; use shares, not absolutes)

| tottime s | cumtime s | ncalls | function |
| --- | --- | --- | --- |
| 7.01 | 6.98 | 31817 | `ISAAC/kit/python/lib/python3.12/json/encoder.py:205(iterencode)` |
| 4.16 | 0.99 | 5635032 | `ISAAC/kit/python/lib/python3.12/copy.py:118(deepcopy)` |
| 3.67 | 5.27 | 34320535 | `src/cascade/control/mobile_base.py:54(<genexpr>)` |
| 2.45 | 1.97 | 111306 | `ISAAC/extscache/omni.warp.core-1.17.0+lx64/warp/_src/context.py:13904(copy)` |
| 2.33 | 8.29 | 805802 | `~:0(<built-in method builtins.any>)` |
| 2.27 | 0.00 | 2 | `ISAAC/extscache/omni.replicator.core-1.13.37+110.4.0.lx64.r.cp312/omni/replicator/core/scripts/orchestrator.py:459(_get_pt_subframes_per_frame)` |
| 2.25 | 2.25 | 16953442 | `~:0(<method 'get' of 'dict' objects>)` |
| 2.09 | 6.49 | 626385 | `src/cascade/control/mobile_base.py:18(_plain_record)` |
| 1.95 | 0.00 | 5 | `ISAAC/extscache/omni.syntheticdata-0.6.17+1066600b.lx64.r.cp312/omni/syntheticdata/scripts/SyntheticData.py:1564(_post_process_graph_tick)` |
| 1.75 | 1.75 | 33704811 | `~:0(<built-in method builtins.ord>)` |
| 1.42 | 0.00 | 24563 | `~:0(<method 'recv' of '_socket.socket' objects>)` |
| 1.35 | 1.35 | 8010291 | `src/cascade/control/mobile_base.py:28(<genexpr>)` |
| 1.35 | 2.73 | 5736960 | `ISAAC/kit/python/lib/python3.12/dataclasses.py:1332(_asdict_inner)` |
| 1.22 | 4.14 | 2026862 | `~:0(<built-in method builtins.all>)` |
| 1.22 | 1.22 | 9260263 | `~:0(<built-in method builtins.getattr>)` |
| 1.14 | 2.52 | 7927758 | `~:0(<built-in method builtins.isinstance>)` |
| 1.06 | 1.62 | 5621760 | `ISAAC/kit/python/lib/python3.12/dataclasses.py:1373(<genexpr>)` |
| 1.06 | 2.42 | 120688 | `ISAAC/extscache/omni.kit.pip_archive-0.0.0+1066600b.lx64.cp312/pip_prebundle/numpy/_core/numeric.py:2373(isclose)` |
| 1.03 | 1.70 | 700600 | `ISAAC/kit/python/lib/python3.12/dataclasses.py:1278(fields)` |
| 1.03 | 5.83 | 11999 | `src/cascade/control/mobile_support.py:74(__post_init__)` |
| 0.99 | 1.24 | 6482 | `src/cascade/control/mobile_support.py:107(as_observation_dict)` |
| 0.85 | 1.62 | 423427 | `ISAAC/extscache/omni.warp.core-1.17.0+lx64/warp/_src/context.py:10272(pack_arg)` |
| 0.75 | 0.00 | 62906 | `~:0(<method 'acquire' of '_thread.lock' objects>)` |
| 0.73 | 0.00 | 15544 | `~:0(<method '_accept' of '_socket.socket' objects>)` |
| 0.71 | 8.64 | 129933 | `ISAAC/kit/python/lib/python3.12/copy.py:217(_deepcopy_dict)` |
| 0.70 | 3.72 | 5751 | `src/cascade/control/newton_bam.py:354(_validate_binding)` |
| 0.69 | 0.69 | 866634 | `~:0(<method 'reduce' of 'numpy.ufunc' objects>)` |
| 0.57 | 0.57 | 2678724 | `src/cascade/control/mobile_support.py:127(<genexpr>)` |
| 0.55 | 0.55 | 6834277 | `~:0(<built-in method builtins.id>)` |
| 0.53 | 3.26 | 23446 | `ISAAC/extscache/omni.warp.core-1.17.0+lx64/warp/_src/context.py:11188(launch)` |
