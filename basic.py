
import numpy as np
import numpy.linalg as la


def transform(points, Rt):
    R = Rt[:3, :3].transpose()
    t = Rt[:3, 3]
    new_points = np.matmul(points, R)
    new_points = new_points + t[np.newaxis, :]
    return new_points

def evaluate(src, ref, Rt):
    src = transform(src, Rt)
    error = ref-src
    mse = la.norm(error, 2)**2 / float(src.shape[0])
    return mse, error
