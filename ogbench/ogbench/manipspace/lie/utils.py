import mujoco
import numpy as np


def get_epsilon(dtype: np.dtype) -> float:
    return {
        np.dtype('float32'): 1e-5,
        np.dtype('float64'): 1e-10,
    }[dtype]


def skew(x: np.ndarray) -> np.ndarray:
    assert x.shape == (3,)
    wx, wy, wz = x
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ]
    )


def mat2quat(mat: np.ndarray):
    """Convert a MuJoCo matrix (9,) to a quaternion (4,)."""
    assert mat.shape == (9,)
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def interpolate(p0, p1, alpha=0.5):
    """Interpolate between two points on a manifold."""
    assert 0.0 <= alpha <= 1.0
    exp_func = getattr(type(p0), 'exp')
    return p0 @ exp_func(alpha * (p0.inverse() @ p1).log())
