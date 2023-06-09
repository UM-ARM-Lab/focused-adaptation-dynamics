import numpy as np
import tensorflow as tf
import torch


def torchify(d, device = "cpu"):
    if isinstance(d, dict):
        return {k: torchify(v, device=device) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.to(device)
    elif isinstance(d, tf.Tensor):
        if d.dtype == tf.string:
            return d.numpy()  # torch doesn't suppor strings in Tensors, so just convert return as a numpy array
        else:
            return torch.from_numpy(d.numpy()).to(device)
    elif isinstance(d, np.ndarray):
        if d.dtype.char in ['U', 'O']:
            return d
        else:
            return torch.from_numpy(d).float().to(device)
    elif isinstance(d, list):
        d0 = d[0]
        if isinstance(d0, dict):
            return [torchify(d_i, device=device) for d_i in d]
        if isinstance(d0, list) or isinstance(d0, np.ndarray) or isinstance(d0, torch.Tensor):
            out_d = [torchify(d_i, device=device) for d_i in d]
            try:
                return torch.stack(out_d)
            except TypeError:
                return out_d
        else:
            return d
    else:
        return d
    raise NotImplementedError()
