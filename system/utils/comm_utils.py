"""Communication accounting for FICL.

Every message is a dict of numbers and arrays. Its size is counted with one fixed rule, written
into every experiment.json, so byte counts are comparable across algorithms:

    floating-point numbers   4 bytes each (fp32 on the wire; the simulation itself computes in fp64)
    integers                 8 bytes each (int64)
    booleans                 1 byte each
    Symmetric(matrix)        only the upper triangle with the diagonal: d (d + 1) / 2 floats

Nested dicts, lists and tuples are counted recursively; None counts as nothing. Strings and other
objects are refused, so nothing travels uncounted.
"""
from __future__ import annotations

import numpy as np

FLOAT_BYTES, INT_BYTES, BOOL_BYTES = 4, 8, 1
WIRE_RULE = {
    "version": 1,
    "float": "4 bytes per number (fp32 on the wire; computation stays fp64)",
    "int": "8 bytes per number (int64)",
    "bool": "1 byte",
    "symmetric_matrix": "upper triangle with the diagonal: d(d+1)/2 floats",
}


class Symmetric:
    """Marks a symmetric matrix (for example a Gram matrix) so only its upper triangle is counted."""

    __slots__ = ("matrix",)

    def __init__(self, matrix):
        shape = tuple(getattr(matrix, "shape", ()))
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"Symmetric needs a square matrix, got shape {shape}")
        self.matrix = matrix


def wire_nbytes(obj) -> int:
    """Bytes of a message under WIRE_RULE."""
    if obj is None:
        return 0
    if isinstance(obj, Symmetric):
        d = int(obj.matrix.shape[0])
        return FLOAT_BYTES * d * (d + 1) // 2
    if isinstance(obj, dict):
        return sum(wire_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(wire_nbytes(v) for v in obj)
    if isinstance(obj, (bool, np.bool_)):
        return BOOL_BYTES
    if isinstance(obj, (int, np.integer)):
        return INT_BYTES
    if isinstance(obj, (float, np.floating)):
        return FLOAT_BYTES
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.bool_:
            return BOOL_BYTES * obj.size
        if np.issubdtype(obj.dtype, np.integer):
            return INT_BYTES * obj.size
        if np.issubdtype(obj.dtype, np.floating):
            return FLOAT_BYTES * obj.size
        raise TypeError(f"cannot count an array of dtype {obj.dtype} in a message")
    if hasattr(obj, "numel") and hasattr(obj, "is_floating_point"):      # torch.Tensor, without importing torch
        n = int(obj.numel())
        if str(obj.dtype) == "torch.bool":
            return BOOL_BYTES * n
        if obj.is_floating_point():
            return FLOAT_BYTES * n
        if not obj.is_complex():
            return INT_BYTES * n
    raise TypeError(f"cannot count a {type(obj).__name__} in a message; send numbers or arrays")


def payload_keys(obj) -> str:
    """Short description of a message for comm.csv, e.g. 'W+b+n'."""
    if isinstance(obj, dict):
        return "+".join(sorted(str(k) for k in obj))
    return type(obj).__name__
