#!/usr/bin/env bash
# Set up the repo on Ubuntu / Debian: system packages, the game files, the emulator, the game host, the renderer and the
# Python environment (.venv) for the training scripts.
#
#   ./setup.sh                          everything (uses sudo for the system packages)
#   ./setup.sh --skip-apt               the system packages are already installed
#   NB_APK_URL=https://host/apk ./setup.sh   download the APK from where you host it (checked by its sha256)
#
# The large files are not in git: the APK (1.17 GB) and the qemu source (129 MB). The APK is put in apk/ by hand or
# downloaded with NB_APK_URL; the qemu source is downloaded from qemu.org.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
APK_NAME="nb_68.279_release_7c5c5152.apk"
APK_SHA256="3c249ea89050220eb32fb7eff0a94f360eb88da19244fa96f51e596bc3dcee43"
QEMU_URL="https://download.qemu.org/qemu-8.2.2.tar.xz"
QEMU_SHA256="847346c1b82c1a54b2c38f6edbd85549edeb17430b7d4d3da12620e2962bc4f3"
APT_PKGS=(build-essential gcc-aarch64-linux-gnu ninja-build pkg-config libegl-dev libgles-dev libx11-dev
          libglib2.0-dev libpixman-1-dev flex bison unzip xz-utils zstd curl)

say() { printf '\n== %s\n' "$*"; }
sha_ok() { echo "$2  $1" | sha256sum -c --status; }

if [ "${1:-}" != "--skip-apt" ]; then
    if command -v apt-get >/dev/null 2>&1; then
        say "system packages"
        sudo apt-get update
        sudo apt-get install -y "${APT_PKGS[@]}"
    else
        echo "Not a Debian/Ubuntu system: install these packages yourself: ${APT_PKGS[*]}"
    fi
fi

say "game file (APK)"
APK="$ROOT/apk/$APK_NAME"
mkdir -p "$ROOT/apk"
if [ -f "$APK" ] && sha_ok "$APK" "$APK_SHA256"; then
    echo "apk/$APK_NAME: ok"
elif [ -n "${NB_APK_URL:-}" ]; then
    echo "downloading the APK from NB_APK_URL ..."
    curl -fL --retry 3 -o "$APK.part" "$NB_APK_URL"
    mv "$APK.part" "$APK"
    sha_ok "$APK" "$APK_SHA256" || { echo "checksum mismatch: the file at NB_APK_URL is not $APK_NAME" >&2; exit 1; }
else
    echo "The APK is missing. Put $APK_NAME into apk/, or rerun with NB_APK_URL=<where you host it>." >&2
    exit 1
fi
if [ ! -f "$ROOT/apk_extract/lib/arm64-v8a/libg.so" ]; then
    say "extracting the APK"
    "$ROOT/loader/setup_apk.sh" "$APK" "$ROOT/apk_extract"
fi

say "emulator (qemu-aarch64-nb)"
if "$ROOT/qemu/bin/qemu-aarch64-nb" --version >/dev/null 2>&1; then
    echo "qemu/bin/qemu-aarch64-nb: ok"
else
    TAR="$ROOT/qemu/src/qemu-8.2.2.tar.xz"
    mkdir -p "$ROOT/qemu/src"
    if [ ! -f "$TAR" ] || ! sha_ok "$TAR" "$QEMU_SHA256"; then
        echo "downloading the qemu 8.2.2 source from qemu.org ..."
        curl -fL --retry 3 -o "$TAR" "$QEMU_URL"
        sha_ok "$TAR" "$QEMU_SHA256" || { echo "checksum mismatch for $TAR" >&2; exit 1; }
    fi
    echo "building the emulator (about 5 minutes) ..."
    bash "$ROOT/qemu/build_qemu_nb.sh"
fi

say "game host (loader/nbhost)"
bash "$ROOT/loader/build.sh"
say "renderer (glhost/nbgl)"
bash "$ROOT/glhost/build.sh"

say "Python environment (.venv)"
if [ ! -x "$ROOT/.venv/bin/python" ]; then python3 -m venv "$ROOT/.venv"; fi
"$ROOT/.venv/bin/pip" install --upgrade pip
"$ROOT/.venv/bin/pip" install -r "$ROOT/requirements.txt"
if "$ROOT/.venv/bin/python" -c "import torch" >/dev/null 2>&1; then
    echo "torch: $("$ROOT/.venv/bin/python" -c 'import torch; print(torch.__version__)')"
else
    echo "torch is not installed. Install the build for your GPU into the venv first: pick it on pytorch.org"
    echo "(CUDA for NVIDIA, ROCm for AMD), run its pip command with .venv/bin/pip, then rerun ./setup.sh --skip-apt"
fi

say "done"
echo "Activate the Python environment with: source .venv/bin/activate"
echo "Then train or evaluate, see README.md"
