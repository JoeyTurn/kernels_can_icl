import math
import os
import numpy as np
import torch


def ensure_numpy(x, dtype=np.float64, clone=False):
    """Convert torch.Tensor to numpy array if necessary.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    result = np.asarray(x, dtype=dtype)
    if clone and result is x:
        return result.copy()
    return result


def ensure_torch(x, dtype=torch.float32, device=None, clone=False):
    """
    Convert array-like to torch.Tensor on the desired device/dtype.

    Notes:
      - In multiprocessing, call torch.cuda.set_device(device_id) in each worker.
        Leaving device=None will then use that worker's GPU via 'cuda'.
      - On single-GPU, device=None -> 'cuda' automatically.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")  # current CUDA device (per-process)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        device = torch.device(device)

    if not isinstance(x, torch.Tensor):
        return torch.as_tensor(x, dtype=dtype, device=device)

    # Already a tensor: move/cast only if needed
    needs_move = (x.device != device) or (x.dtype != dtype)
    if needs_move:
        x = x.to(device=device, dtype=dtype, non_blocking=True)

    return x.clone() if clone else x


def available_memory_bytes():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # current CUDA device (per-process)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda" and torch.cuda.is_available():
        cuda_device = device
        if cuda_device.index is None:
            cuda_device = torch.device("cuda", torch.cuda.current_device())
        free, _total = torch.cuda.mem_get_info(cuda_device)
        return int(free)

    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * avail_pages)
    except Exception:
        return None


def estimate_max_n(*,config=None, n = None, max_n = None, min_n = None,
    memory_safety = 0.5, matrix_factor: float = 4.0,):
    if config is not None:
        if n is None:
            n = getattr(config, "n", None)
        if max_n is None:
            max_n = getattr(config, "max_n", None)
        if min_n is None:
            min_n = getattr(config, "min_n", None)
        memory_safety = float(getattr(config, "memory_safety", memory_safety))
        matrix_factor = float(getattr(config, "matrix_factor", matrix_factor))

    if n is not None:
        return int(n)

    avail = available_memory_bytes()
    if avail is None:
        return int(max_n) if max_n is not None else None

    bytes_per = torch.tensor([]).element_size()
    n_est = int(math.floor(math.sqrt((avail * float(memory_safety))/ (bytes_per * float(matrix_factor)) )))

    if max_n is not None:
        n_est = min(n_est, int(max_n))
    if min_n is not None:
        n_est = max(n_est, int(min_n))
    return max(n_est, 1)


def seed_everything(seed_int, device_id=0, make_generators=True, deterministic=False):
    """
    Set Python/NumPy/Torch (CPU & current CUDA device) RNGs.
    Returns a torch.Generator() seeded the same way if requested.
    """
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_int)  # current device
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return (torch.Generator(device=f"cuda:{device_id}").manual_seed(seed_int), np.random.default_rng(seed_int)) if make_generators else None

