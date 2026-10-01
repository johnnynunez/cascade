# Independent RGB and depth encoding

An RGB-only read of an Isaac frame previously encoded both JPEG and compressed
depth. The observer requests RGB hashes without requesting a depth payload,
so that read performed unused depth work. `_LazyFrame` now materializes each
component independently. This changes encoding work after capture; it leaves
the [render/history binding](isaac-render-frame-history.md) intact.

## Measured observer cost

Passive profile 10 ran source `6e1bc81a` on the local RTX Pro workstation.
Its [original observer receipt](evidence/isaac-profile10/observer.json) records
100 complete samples in 24.697 seconds, with no actuator commands or native
task. Original geometry, inventory, camera identity and freshness checks
passed. All observed physics steps advanced, CUDA attestations remained valid,
and each camera supplied 50 distinct capture timestamps and JPEG hashes.

The [independent analysis](evidence/isaac-profile10/analysis.json) measured
15.424 seconds of host elapsed time inside the observer payload: the frame-hash section accumulated
4.842 seconds and the camera-hash section 7.297 seconds. Samples with a new
capture had a median payload duration of 272.239 ms; repeated captures had
33.435 ms. These sections include RGB lookup, base64 decoding and hashing;
they do not isolate JPEG, depth compression or hashing costs individually.
Source inspection establishes the unnecessary depth work in the former
RGB-only path. These passive measurements do not attribute every execution
job in the active profile 09 trace.

[Administrative closure](evidence/isaac-profile10/administrative-close.json)
stopped the owned services after the observer joined with no pending RPC.
The [root check](evidence/isaac-profile10/close-root-review.json) verified all
1,277 source inputs and 12 protected process identities. This is passive
measurement and closure evidence, without a manipulation result.

## Passive repeat on the candidate

Profile 11 ran combined source `d1d54dc` with the same runtime parameters.
Its [observer receipt](evidence/isaac-profile10/observer11.json) records 100
samples in 15.186 seconds, with 50 new and 50 repeated captures. Each of the
three cameras supplied 50 distinct capture timestamps and JPEG hashes.
Geometry, inventory, original camera freshness/identity and physics checks
passed. No actuator commands or native phases were present.

The [source-bound comparison](evidence/isaac-profile10/comparison10-11.json)
uses host elapsed section timings, with all samples retained:

| Measurement | Profile 10 (`6e1bc81a`) | Profile 11 (`d1d54dc`) |
|---|---:|---:|
| New-capture payload median, 50 samples each | 272.239 ms | 68.170 ms |
| New-capture payload p95 | 285.597 ms | 83.892 ms |
| Repeated-capture payload median, 50 samples each | 33.435 ms | 30.376 ms |
| New-capture frame-hash median | 95.995 ms | 10.369 ms |
| New-capture camera-hashes median | 144.807 ms | 21.410 ms |
| New-capture geometry median | 21.193 ms | 21.514 ms |
| Payload elapsed sum, 100 samples | 15.424 s | 5.380 s |

A [retained slow sample](evidence/isaac-profile10/comparison10-11-arithmetic.json)
took 309.668 ms, including 267.636 ms in geometry. That identifies the timed
section, without establishing an internal USD/SDK cause or authorizing geometry
caching. This comparison has one fresh passive scene per source, with no
repeated-run confidence interval, absolute GPU timing or native task result.
New captures are defined within each run; frames are not paired across runs.

[Owned closure](evidence/isaac-profile10/administrative-close11.json) completed
after the observer joined with no pending RPC. The
[root check](evidence/isaac-profile10/close11-root-review.json) verified 1,305
source inputs, 12 protected process identities and closure of the owned ports.
This measurement does not resolve profile 09's failed placement or NV15's
release/retreat failure.

## Encoding contract

- RGB and depth each have their own lock and successful-result cache. Reading
  RGB does not acquire the depth lock; metadata reads start neither encoder.
- A successful component is encoded once per retained frame. Its raw buffer
  is released after publication; unused components remain with the frame until
  requested or discarded. Codec failures propagate and retain that component
  for a later read, without re-encoding a previously successful component.
- `wire()` still materializes the complete packet and preserves metadata,
  field order, JPEG quality 90, depth zlib level 3, nonfinite-depth sanitation
  and null depth when absent. Existing depth consumers receive the same data.
- Capture timestamps, epochs, proprioception, buffer copies, frame history,
  camera cadence, physics steps, motion deadlines and safety gates are unchanged.
  Encoding completion never refreshes a capture's age.

The [focused evidence](evidence/isaac-profile10/codec-focused.json) retains
three expected baseline failures, 16 new test successes and 123 existing
regression successes. Tests exercise component access, unchanged complete
serialization, concurrent readers, a blocked depth encoder and codec failures.
An [independent peer](evidence/isaac-profile10/codec-peer.json) verified that
only the `_LazyFrame` class changed in the bridge's executable syntax tree.

## CPU replay and validation limits

A separate [CPU comparison](evidence/isaac-profile10/cpu-codec-comparison.json)
used one retained camera capture, whose RGB pixels had already been JPEG-decoded.
Both versions received identical inputs. Twenty trials per version and mode
alternated execution order; the values below are median host elapsed times.

| Request | Previous encoding | Independent components |
|---|---:|---:|
| RGB only | 44.323 ms | 2.196 ms |
| Complete `wire()` packet | 44.103 ms | 44.108 ms |

RGB reads made zero depth-encoder calls in the candidate. Complete wire
serialization, RGB bytes and capture metadata matched exactly. The replay used
CPU codecs, with no Kit, GPU or RPC. It establishes the measured benefit for
this RGB-only input; it does not establish a whole-task speedup or continuous
camera freshness under live load.

Product commit `8935c0e` is included in combined candidate `d1d54dc`, alongside
the separately tested release-open stability change and profile 09 docs.
Its [full suite](evidence/isaac-profile10/combined-full.json) passed
[3,757 tests](evidence/isaac-profile10/combined-test-summary.json), with 43
skipped and four deselected in 339.27 seconds; all 1,305 bound inputs stayed
unchanged. The passive repeat on that source passed as recorded above. Native
placement, release, home, camera, campaign and restart checks remain pending.
The previous physical failures are retained.
The [input index](evidence/isaac-profile10/retained-inputs.json) keeps each
measurement and software result bound to its own source.
