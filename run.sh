#!/usr/bin/env bash
# Launcher for Project Ecosystem.
# Sets up the environment (esp. libEGL on minimal systems) and starts the app.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Make sure the project root is on PYTHONPATH so `python -m ecosystem.app`
# works regardless of where the script is invoked from.
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# If libEGL.so.1 isn't on the system loader path, look for a bundled copy
# under scripts/lib/ (see README for how to populate it). This makes the
# app work on minimal/headless Linux installs.
if [ ! -f /usr/lib/x86_64-linux-gnu/libEGL.so.1 ] && [ -d "$SCRIPT_DIR/scripts/lib" ]; then
    export LD_LIBRARY_PATH="$SCRIPT_DIR/scripts/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Honour user overrides.
if [ -n "${QT_QPA_PLATFORM:-}" ]; then
    :
elif [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    echo "No DISPLAY or WAYLAND_DISPLAY detected; falling back to offscreen platform." >&2
    export QT_QPA_PLATFORM=offscreen
fi

exec python3 -m ecosystem.app "$@"
