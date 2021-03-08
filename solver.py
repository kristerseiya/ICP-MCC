
import numpy as np
import numpy.linalg as la
from numpy import newaxis
from typing import Union

def kabsch_umeyama(src: np.ndarray,
				   ref: np.ndarray,
				   with_scaling: bool = False) -> np.ndarray:

	if src.ndim != 2 or ref.ndim != 2:
		raise ValueError("Arguments must have shape of 2")

	if src.shape != ref.shape:
		raise ValueError("source and reference must have the same shape")


	n = src.shape[0]
	dim = src.shape[1]

	src_mean = src.mean(0)
	ref_mean = ref.mean(0)

	src_demean = src - src_mean[newaxis, :]
	ref_demean = ref - ref_mean[newaxis, :]

	src_var = la.norm(src_demean, 2) ** 2 / n

	sigma = np.matmul(src_demean.transpose(), ref_demean)

	u, s, vh = la.svd(sigma)

	Rt = np.eye(dim+1, dim+1)

	S = np.ones(dim)
	if la.det(u) * la.det(vh) < 0:
		S[-1] = -1

	R = np.matmul(np.matmul(u, np.diag(S)), vh)
	Rt[:dim, :dim] = R

	if with_scaling:
		c = s.dot(S) / src_var
		Rt[dim, :dim] = ref_mean
		Rt[dim, :dim] -= c * np.matmul(src_mean, R)
		Rt[dim, :dim] *= c
	else:
		Rt[dim, :dim] = ref_mean
		Rt[dim, :dim] -= np.matmul(src_mean, R)

	return Rt.transpose()

def calc_interquartile_range(x):
	x = x.reshape([-1])
	x = np.sort(x)
	low = len(x) // 4
	high = len(x) // 4 * 3
	assert(x[high] >= x[low])
	return x[high] - x[low]


def kabsch_umeyama_mcc(src: np.ndarray,
				   	  ref: np.ndarray) -> np.ndarray:

	if src.ndim != 2 or ref.ndim != 2:
		raise ValueError("Arguments must have shape of 2")

	if src.shape != ref.shape:
		raise ValueError("source and reference must have the same shape")


	n = src.shape[0]
	dim = src.shape[1]

	error2 = la.norm(ref - src, 2, axis=-1, keepdims=True)**2
	sigmaE = np.std(error2)
	interq = calc_interquartile_range(error2)
	# sigma2 = 1.06 * min(sigmaE, interq / 1.354) * np.power(n, -0.25)
	sigma2 = 1
	ke = np.exp(-error2/(2*sigma2))
	p = src - ke * src / ke.sum()
	q = ref - ke * ref / ke.sum()
	H = ke * q
	H = np.matmul(p.transpose(), H)

	u, d, vh = la.svd(H)

	T = np.eye(dim+1, dim+1)

	D = np.ones(dim)
	if la.det(u) * la.det(vh) < 0:
		D[-1] = -1

	R = np.matmul(np.matmul(u, np.diag(D)), vh)
	T[:dim, :dim] = R

	t = ref - np.matmul(src, R)
	t = (ke * t).sum(axis=0) / ke.sum()
	T[dim, :dim] = t

	return T.transpose()
