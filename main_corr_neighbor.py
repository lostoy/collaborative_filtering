from dataset import load_train_test
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse
from evaluation import get_recall_precision_N, get_rmse
def similarity_fn(R, knn):
    '''
    :param R: m x d
    :return: sim: m x knn, ind: m x knn
    '''
    R = normalize(R)
    m = R.shape[0]
    sims = []
    inds = []
    batch = 5000
    for i in range(0, m, batch):
        t_sim = (R[i:i+batch].dot(R.transpose())).reshape(-1, m) # 1 x m
        t_ind = np.argsort(-t_sim, 1)[:, 1:knn+1]
        t_sim = -np.sort(-t_sim, 1)[:, 1:knn+1]
        sims.append(t_sim)
        inds.append(t_ind)
    sims = np.concatenate(sims, 0)
    inds = np.concatenate(inds, 0)
    return sims, inds


def main():
    N = 10
    knn_list = [25, 50, 100, 200, 400]
    train_R, test_R = load_train_test()
    m, n = train_R.shape
    train_items = np.split(train_R.indices, train_R.indptr[1:-1])


    train_R = train_R.toarray()
    mean_u = train_R.mean(1).reshape(-1, 1)
    mean_q = train_R.mean(0).reshape(1, -1)
    for i in range(n):
        train_R[train_R[:, i] == 0, i] = mean_q[0, i]
    train_R -= mean_u
    train_R /= np.maximum(np.std(train_R, 1, keepdims=True), 1e-3)

    sims, sim_inds = similarity_fn(train_R, max(knn_list))
    for cur_R, cur_set in zip((test_R, ), ('test', )):
        for knn in knn_list:
            recalls = []
            precs = []
            rmses = []

            for u_id in range(m):
                knn_scores = train_R[sim_inds[u_id, :knn]]
                #knn_means = mean_u[sim_inds[u_id, :knn]]
                knn_sims = sims[u_id, :knn]
                t_scores = (
                np.diag(knn_sims / np.maximum(np.sum(knn_sims), 1e-3)).dot(knn_scores)).mean(
                    0).reshape(-1, n)  # 1 x n
                t_scores = t_scores + mean_u[u_id]

                t_scores[:, train_items[u_id]] = -10.0
                t_scores = np.asarray(t_scores)

                t_rec, t_prec = get_recall_precision_N(t_scores, cur_R[u_id], N=N)
                t_rmse = get_rmse(t_scores, cur_R[u_id])

                recalls.append(t_rec)
                precs.append(t_prec)
                rmses.append(t_rmse)

            print("cur_set: {}\t"
                  "Knn: {} \t "
                  "recall: {} \t"
                  "prec: {} \t"
                  "rmse: {}".format(cur_set, knn, np.mean(recalls), np.mean(precs), np.mean(rmses)))
        print('--------------')
if __name__ == '__main__':
    main()
