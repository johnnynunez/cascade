# DGX Spark portability

The Brev kitchen container targets x86_64 and RTX PRO 6000 Blackwell.
DGX Spark uses arm64 and GB10; do not reuse the Brev binary image or assume
its separate VRAM limits describe Spark's shared memory.

The scene, CASCADE tool API, guide, extension and source-level health checks
can be shared. Provision an NVIDIA-supported arm64 Isaac installation,
a native model server compiled for GB10, and separate memory/rendering
limits before testing the same tool and physics contracts.

The Spark presenter installer uses PhysX CUDA, real GraspGen-X and
Qwen3.8-27B Q4 with its vision projector. This Brev deployment uses
Qwen3.8-27B Q8_0. Neither Brev results nor local configuration tests
certify a Spark cold installation or its native-agent placement proof.
Use [DGX Spark setup](DGX_SPARK_SETUP.md) for the pinned Spark source and
current validation status; Cosmos, nvblox and JEv are not required for that path.
