from dataset import load_train_test
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse.linalg
from evaluation import get_rmse, get_recall_precision_N

def main():
    N = 10
    k_dim_list = [2, 5, 10, 25, 50, 75, 90]
    train_R, test_R = load_train_test()
    m, n = train_R.shape
    train_items = np.split(train_R.indices, train_R.indptr[1:-1])

    train_R = train_R.toarray()
    mean_u = train_R.mean(1).reshape(-1, 1)
    mean_q = train_R.mean(0).reshape(1, -1)
    for i in range(n):
        train_R[train_R[:, i] == 0, i] = mean_q[0, i]
    train_R -= mean_u

    #U, S, Vt = scipy.sparse.linalg.svds(train_R, max(k_dim_list))
    U, S, Vt = scipy.linalg.svd(train_R, False)
    P = U*S.reshape(1, -1) # m x d
    # Vt

    for cur_R, cur_set in zip((test_R, ), ('test', )):
        for k_dim in k_dim_list:
            recalls = []
            precs = []
            rmses = []

            for u_id in range(m):
                #t_p = U[u_id, -k_dim-1:].reshape(1, -1) # 1 x kdim
                t_p = P[u_id, :k_dim].reshape(1, -1)  # 1 x kdim
                #t_Q = Vt[-k_dim - 1:, :]
                t_Q = Vt[:k_dim, :]
                t_scores = mean_u[u_id] + t_p.dot(t_Q)
                t_scores[:, train_items[u_id]] = -10.0
                t_scores = np.asarray(t_scores)

                t_rec, t_prec = get_recall_precision_N(t_scores, cur_R[u_id], N=N)
                t_rmse = get_rmse(t_scores, cur_R[u_id])

                recalls.append(t_rec)
                precs.append(t_prec)
                rmses.append(t_rmse)

            print("cur_set: {}\t"
                  "k_dim: {} \t "
                  "recall: {} \t"
                  "prec: {} \t"
                  "rmse: {}".format(cur_set, k_dim, np.mean(recalls), np.mean(precs), np.mean(rmses)))
            print('--------------')

if __name__ == '__main__':
    main()
