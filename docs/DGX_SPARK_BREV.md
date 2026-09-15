# DGX Spark portability

The Brev kitchen container targets x86_64 and RTX PRO 6000 Blackwell.
DGX Spark uses arm64 and GB10; do not reuse the Brev binary image or assume
its separate VRAM limits describe Spark's shared memory.

The scene, CASCADE tool API, guide, extension and source-level health checks
can be shared. Provision an NVIDIA-supported arm64 Isaac installation,
a native model server compiled for GB10, and separate memory/rendering
limits before testing the same tool and physics contracts.

Upstream defaults to Cosmos3-Edge on Spark. This Brev deployment uses
Qwen3.8-27B Q8_0. Neither the Brev results nor local configuration tests
certify Cosmos, Spark cold start or a physical pick on Spark.
