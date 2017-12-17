import scipy.sparse
import numpy as np
def get_rmse(predict_R, gt_R):
    rows, cols, gt_vals = scipy.sparse.find(gt_R)
    if len(cols) == 0:
        return 0.0
    return np.sqrt(np.mean(np.square((predict_R[rows, cols] - gt_vals))))

def get_recall_precision_N(predict_R, gt_R, N=10):
    '''

    :param predict_R: predict_R only contains prediction of items not in train set
    :param gt_R: gt stores in sparse format
    :param N:
    :return:
    '''
    m, n = gt_R.shape
    sort_inds = np.argpartition(-predict_R, N, axis=1)[:, :N] # m x N
    gt_R = gt_R.copy()
    gt_R.data *= gt_R.data > 3.0
    gt_R.eliminate_zeros()
    gt_inds = np.split(gt_R.indices, gt_R.indptr[1:-1]) # m x ?
    len_gt = np.asarray(map(len, gt_inds), dtype=np.float)

    intersects = np.asarray(map(lambda p_gt_inds: len(np.intersect1d(p_gt_inds[0], p_gt_inds[1])), zip(sort_inds, gt_inds)),
                            dtype=np.float) # m x 1
    recall = intersects / np.maximum(len_gt, 1e-3)
    recall[len_gt == 0] = 1
    recall = np.mean(recall)
    precision = np.mean(intersects / N)

    return recall, precision
