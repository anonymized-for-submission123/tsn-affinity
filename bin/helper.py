import numpy as np
import torch


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _seed_signature(seed: int) -> str:
    return f"__s{int(seed)}"