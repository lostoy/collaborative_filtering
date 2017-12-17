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
    for i in range(m):
        t_sim = (R[i].dot(R.transpose())).toarray().reshape(m) # 1 x m
        t_ind = np.argsort(-t_sim)[1:knn+1]
        t_sim = t_sim[t_ind]

        sims.append(t_sim)
        inds.append(t_ind)
    sims = np.array(sims)
    inds = np.array(inds)
    return sims, inds

def main():
    N = 10
    knn_list = [25, 50, 100, 200, 400]
    train_R, test_R = load_train_test()
    mean_u = scipy.sparse.csr_matrix.mean(train_R, 1)
    m, n = train_R.shape
    train_items = np.split(train_R.indices, train_R.indptr[1:-1])
    sims, sim_inds = similarity_fn(train_R, max(knn_list))

    for cur_R, cur_set in zip((test_R, ), ('test', )):
        for knn in knn_list:
            recalls = []
            precs = []
            rmses = []

            for u_id in range(m):
                knn_scores = train_R[sim_inds[u_id, :knn]]
                knn_means = mean_u[sim_inds[u_id, :knn]]
                knn_sims = sims[u_id, :knn]
                t_scores = (
                scipy.sparse.diags(knn_sims / np.maximum(np.sum(knn_sims), 1e-3)) * (knn_scores - knn_means)).mean(
                    0)  # 1 x n
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
