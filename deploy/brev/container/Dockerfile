FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43 AS llama
ARG LLAMA_REVISION=5f436dddb440a288ee5611d7d1eca564a6aca9f4
ARG CUDA_ARCHITECTURES=120a-real
RUN apt-get update && apt-get install -y --no-install-recommends git cmake ninja-build libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git init /build && cd /build && git remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git fetch --depth=1 origin "$LLAMA_REVISION" && git checkout --detach FETCH_HEAD \
    && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
       -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" -DLLAMA_BUILD_TESTS=OFF \
       -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
       -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --target llama-server -j 6

FROM node:24.16.0-bookworm-slim@sha256:ca520832af80fa37a57c14077ed0fcdd83b5aefccc356059fdc3a9a05b78ae1f AS node

FROM nvcr.io/nvidia/isaac-sim:6.1.0@sha256:0c16dd67d09a70ea474f2c809a13c4d09bd23c738184cf1bf28af487ecd39080
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv ca-certificates libssl3t64 libgomp1 libglib2.0-0t64 libgl1 zstd \
    && rm -rf /var/lib/apt/lists/*
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/LICENSE /opt/paai/node/LICENSE
COPY --from=llama /build/build/bin/ /opt/paai/llama/
COPY --from=llama /build/LICENSE /opt/paai/llama/LICENSE
COPY --from=llama /usr/local/cuda/lib64/libcublas.so* /opt/paai/cuda/lib/
COPY --from=llama /usr/local/cuda/lib64/libcublasLt.so* /opt/paai/cuda/lib/
COPY --from=llama /usr/local/cuda/lib64/libcudart.so* /opt/paai/cuda/lib/
COPY requirements.txt /opt/paai/requirements.txt
COPY clip-source.tar.gz /opt/paai/clip-source.tar.gz
RUN python3.12 -m venv /opt/paai/venv \
    && /opt/paai/venv/bin/pip install --no-cache-dir --timeout 60 \
       --index-url https://download.pytorch.org/whl/cu130 torch==2.14.0+cu130 torchvision==0.29.0+cu130 \
    && /opt/paai/venv/bin/pip install --no-cache-dir --timeout 60 -r /opt/paai/requirements.txt \
    && printf '/workspace/src\n' > /opt/paai/venv/lib/python3.12/site-packages/paai-source.pth
COPY openclaw.tar.zst /tmp/openclaw.tar.zst
RUN echo '73c595199cd2a4cc46e503e09101318f09c21ca878596f662a540d32cd7243bb  /tmp/openclaw.tar.zst' | sha256sum -c - \
    && mkdir -p /opt/paai/openclaw \
    && tar --zstd -xf /tmp/openclaw.tar.zst --strip-components=2 -C /opt/paai/openclaw \
    && rm /tmp/openclaw.tar.zst
COPY entrypoint.py /opt/paai/entrypoint.py
COPY materialize_ui.mjs /opt/paai/materialize_ui.mjs
COPY --from=llama /usr/lib/x86_64-linux-gnu/libnccl.so* /opt/paai/cuda/lib/
WORKDIR /workspace
ENV PATH=/opt/paai/venv/bin:/usr/local/bin:/usr/bin:/bin
USER 1000:1000
ENTRYPOINT ["/opt/paai/venv/bin/python", "/opt/paai/entrypoint.py"]
