FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    WEBOTS_HOME=/usr/local/webots \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    xvfb \
    python3 \
    python3-pip \
    libgl1 \
    libxcomposite1 \
    libxrandr2 \
    libxi6 \
    libxtst6 \
    libnss3 \
    && rm -rf /var/lib/apt/lists/*

ARG WEBOTS_VERSION=R2023b
RUN wget -q "https://github.com/cyberbotics/webots/releases/download/${WEBOTS_VERSION}/webots-${WEBOTS_VERSION}-x86-64.tar.bz2" \
    -O /tmp/webots.tar.bz2 && \
    tar -xjf /tmp/webots.tar.bz2 -C /usr/local && \
    mv /usr/local/webots-* "${WEBOTS_HOME}" 2>/dev/null || true && \
    rm /tmp/webots.tar.bz2

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY simulation/ ./simulation/
COPY robots/ ./robots/
COPY rl/ ./rl/
COPY coordination/ ./coordination/
COPY config/ ./config/
COPY data/ ./data/
COPY scripts/start_simulation.py ./scripts/start_simulation.py

ENV PATH="${WEBOTS_HOME}:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}"

# Headless launch: Xvfb provides a virtual display since Webots
# expects one even with rendering disabled via --no-rendering.
ENTRYPOINT ["xvfb-run", "--auto-servernum", "--server-args=-screen 0 1024x768x24"]
CMD ["webots", "--batch", "--no-rendering", "--stdout", "--stderr", \
     "/app/simulation/webots/worlds/maze.wbt"]
