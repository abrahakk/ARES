FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    WEBOTS_HOME=/usr/local/webots \
    PYTHONUNBUFFERED=1 \
    LIBGL_ALWAYS_SOFTWARE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget bzip2 ca-certificates libglu1-mesa libopenal1 libglib2.0-0 libx11-6 libxext6 libxrender1 libxcursor1 libxdamage1 libxfixes3 libxss1 libx11-xcb1 libxcb-glx0 libxcb-randr0 libxcb-shm0 libxcb-sync1 libxcb-xfixes0 libxcb-xinput0 libxkbfile1 libegl1 libglx-mesa0 libnspr4 libxslt1.1 libxml2 libpng16-16 libfreetype6 libssl3 libpulse0 libdrm2 libgbm1 libsm6 libice6 libxkbcommon-x11-0 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 libxcb-cursor0 libgl1-mesa-dri libasound2 libfontconfig1 libdbus-1-3 libxi6 libxrandr2 libxcomposite1 libxtst6 libnss3 libjpeg-turbo8 \
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
RUN pip3 install --no-cache-dir --default-timeout=300 --retries 10 torch --index-url https://download.pytorch.org/whl/cpu && pip3 install --no-cache-dir --default-timeout=300 --retries 10 -r requirements.txt

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
