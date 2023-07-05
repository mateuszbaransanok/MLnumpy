import numpy as np


def compute_output_shape(
    input_size: int,
    kernel_size: int,
    pad_size: int,
    stride_size: int,
) -> int:
    return int((input_size - kernel_size + 2 * pad_size) / stride_size) + 1


def add_padding(
    array: np.ndarray,
    h_pad: int,
    w_pad: int,
) -> np.ndarray:
    return np.pad(
        array=array,
        pad_width=((0, 0), (h_pad, h_pad), (w_pad, w_pad), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def remove_padding(
    array: np.ndarray,
    h_pad: int,
    w_pad: int,
) -> np.ndarray:
    if h_pad:
        return array[:, h_pad:-h_pad]
    if w_pad:
        return array[:, :, w_pad:-w_pad]
    return array
