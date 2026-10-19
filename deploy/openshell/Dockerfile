# cascade's OpenClaw agent image for a standalone OpenShell 0.1.x sandbox.
# Built by `scripts/openshell_rig.py build`; see docs/OPENSHELL.md.
#
# Two Landlock findings from the live spike are baked in:
#  - OpenShell's default filesystem policy lets the agent read /usr and write
#    /tmp, so the config template lives under /usr/local/share (a copy under
#    /opt was "Permission denied" and the sandbox went to phase Error);
#  - SQLite (OpenClaw's session store) needs its temp files in /tmp.
#
# OpenClaw 2026.9.3 needs node >= 24.16; the base is pinned by digest.
FROM node:24-bookworm-slim@sha256:d6aa754f16b3197301076f047b5def2f02ea1dbbc2ca920407d46d7ec7f87b20

ARG OPENCLAW_VERSION=2026.9.3

# OpenShell's supervisor needs iproute2 in the image; curl is for probes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git iproute2 \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g --no-audit --no-fund "openclaw@${OPENCLAW_VERSION}" \
    && openclaw --version

COPY openclaw.json /usr/local/share/cascade/openclaw.json
COPY render-config.mjs /usr/local/share/cascade/render-config.mjs
# The Franka FR3 agent profile (CASCADE_AGENT_PROFILE=franka): the operator
# command plugin and its instructions, read-only next to the template.
COPY franka-plugin /usr/local/share/cascade/franka-plugin
# The Cosmos3-Edge agent profile (CASCADE_AGENT_PROFILE=cosmos): its prompt
# plugin (short system prompt, user turns without OpenClaw's JSON envelopes).
COPY cosmos-plugin /usr/local/share/cascade/cosmos-plugin
COPY --chmod=0755 cascade-openclaw /usr/local/bin/cascade-openclaw

RUN install -d -o node -g node -m 0700 /sandbox

ENV HOME=/sandbox \
    TMPDIR=/tmp \
    SQLITE_TMPDIR=/tmp
WORKDIR /sandbox
USER node
CMD ["cascade-openclaw"]
