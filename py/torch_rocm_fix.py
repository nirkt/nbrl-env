"""Import before using torch.cuda on this host (see work/NOTES.md T5).

torch 2.11.0+rocm7.13.0 (user site-packages, bundled ROCm 7.13 'TheRock' libraries) counts ROCm devices through AMD-SMI
first (torch.cuda._device_count_amdsmi) and only falls back to HIP when that returns < 0. The `amdsmi` module it
imports is the SYSTEM one (/usr/local/lib/python3.12/dist-packages/amdsmi 26.2.2, built for ROCm 7.2.1); after torch
has loaded its own ROCm 7.13 libraries into the process, that module enumerates 0 processors (it sees 1 without torch).
torch then reports device_count() == 0 although HIP itself (hipGetDeviceCount) sees the RX 9070 XT and kernels run.
This forces the HIP-based count. Nothing on the system is changed."""
import torch
import torch.cuda as _tc

if torch.version.hip and _tc.is_available():
    _tc._device_count_amdsmi = lambda: -1
    _tc._cached_device_count = None

if __name__ == "__main__":
    print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.device_count(),
          torch.cuda.get_device_name(0) if torch.cuda.device_count() else None)

# learn round: our checkpoints store numpy scalars (metrics, args); torch >= 2.6 defaults torch.load to
# weights_only=True, which refuses them. Only this project's own files are loaded, so default to False.
_orig_torch_load = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load
