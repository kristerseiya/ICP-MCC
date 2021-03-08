
from sklearn.neighbors import KDTree
import solver
import numpy as np
import numpy.linalg as la
import basic
import matplotlib.pyplot as plt


def icp(pcd1, pcd2, max_dist, mode='MSE',
        bidirectional=False,
        init=None, iter=30,
        real_T=None, verbose=False):
    if init is None:
        init = np.eye(pcd1.shape[-1]+1)

    T = init
    result = init
    pcd1 = basic.transform(pcd1, init)
    kdt2 = KDTree(pcd2)
    if bidirectional:
        kdt1 = KDTree(pcd1)
    for i in range(iter):

        dist, idx = kdt2.query(pcd1)
        dist = dist.squeeze(-1)
        idx = idx.squeeze(-1)
        src = pcd1[dist <= max_dist]
        ref = pcd2[idx[dist <= max_dist]]
        if bidirectional:
            dist, idx = kdt1.query(pcd2)
            dist = dist.squeeze(-1)
            idx = idx.squeeze(-1)
            src = np.concatenate([src, pcd1[idx[dist <= max_dist]]], axis=0)
            ref = np.concatenate([ref, pcd2[dist <= max_dist]], axis=0)
        if mode == 'MSE':
            T = solver.kabsch_umeyama(src, ref)
        elif mode == 'MCC':
            T = solver.kabsch_umeyama_mcc(src, ref)

        if real_T is not None:
            ideal_T = np.matmul(real_T, la.inv(result))
            _, true_error = basic.evaluate(src, ref, ideal_T)
            if true_error.shape[0] > 1000:
                show_idx = np.random.choice(true_error.shape[0], size=(1000,), replace=False)
                true_error = true_error[show_idx]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(true_error[:, 0], true_error[:, 1], true_error[:, 2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()

        pcd1 = basic.transform(pcd1, T)
        result = np.matmul(T, result)

        if verbose:
            print('Iteration #{:d}'.format(i+1))
            mse, _ = basic.evaluate(src, ref, T)
            print('MSE: {:.4f}'.format(mse))
            print('Pairs: {:d}\n'.format(src.shape[0]))

    return result
