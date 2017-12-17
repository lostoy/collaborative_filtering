from dataset import load_train_test
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse.linalg
from evaluation import get_rmse, get_recall_precision_N

def main():
    N = 10
    k_dim = 10
    update_percent_list = [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
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

    # Vt

    for cur_R, cur_set in zip((test_R, ), ('test', )):
        for update_percent in update_percent_list:
            recalls = []
            precs = []
            rmses = []
            n_update = int(m * update_percent * 1.0 / 100)
            n_batch = m - n_update
            train_R_batch = train_R[:n_batch]
            train_R_update = train_R[n_batch:]
            U_batch, S, Vt = scipy.linalg.svd(train_R_batch, False)

            U_batch = U_batch[:, :k_dim]
            S = S[:k_dim]
            Vt = Vt[:k_dim, :]

            P_batch = U_batch * S.reshape(1, -1)  # m x d
            P_update = train_R_update.dot(Vt.transpose())
            P = np.concatenate((P_batch, P_update), 0)

            for u_id in range(m):
                #t_p = U[u_id, -k_dim-1:].reshape(1, -1) # 1 x kdim
                t_p = P[u_id, :].reshape(1, -1)  # 1 x kdim
                #t_Q = Vt[-k_dim - 1:, :]
                t_Q = Vt
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
                  "update_percent: {}\t"
                  "recall: {} \t"
                  "prec: {} \t"
                  "rmse: {}".format(cur_set, k_dim, update_percent, np.mean(recalls), np.mean(precs), np.mean(rmses)))


if __name__ == '__main__':
    main()
