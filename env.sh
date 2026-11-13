# source this in WSL (Ubuntu-24.04) before any rl/ script: points nirkt's MAEnv at our libg 68 host
B=$HOME/nb68/bulletlabels
export RL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this folder, wherever it is checked out
export NBRL=$(dirname "$RL_ROOT")/reference/nirkt/nbrl-env
export NBHOST_BIN=$B/native/loader/nbhost_rl   # = nbhost_maps + the rl round (team groups, obs fields); build: README
export NBHOST_APK=$B/game
export NB_PIX_CAP=10368000   # = 2400*1080*4, the frame slot nbhost_maps is built with (a mismatch kills every boot silently)
export NBGL_BIN=$B/native/glhost/nbgl
export NBPC_PAYLOADS=$B/native/payloads
q=$(ls $B/native/qemu/bin/qemu-aarch64-nb $HOME/v5dl/qemu-nb/qemu-aarch64-nb 2>/dev/null | head -1)
export NBHOST_QEMU=$(readlink -f "$q")
export PYTHONPATH=$RL_ROOT:$B/vendor:$NBRL/py:$NBRL/train   # vendor nbpc first: ours has team_mask / map_location and the host PIX_CAP
source $HOME/nb68/venv/bin/activate
mkdir -p $HOME/nb68/rl
