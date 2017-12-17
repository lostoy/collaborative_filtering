import numpy as np
import pickle
import os
import scipy.sparse

p_train = 0.7

def load_train_test():
    if os.path.exists('./data/train_test.p'):
        with open('./data/train_test.p', 'rb') as f:
            R_train, R_test = pickle.load(f)
            # R_train = R_train[:1000]
            # R_test = R_test[:1000]
        return R_train, R_test

    rows = []
    n_rating = 0
    with open('./data/jester-data-1.csv', 'r') as f:
        for row in f:
            row = row.strip()
            row = row.split(',')
            n_rating += int(row[0])
            row = [float(r) for r in row[1:]]
            rows.append(row)
    R = np.asarray(rows)
    m, n = R.shape
    print('load raw R data with size: {}'.format(R.shape))
    print('# of ratings: {}, sparsity: {}'.format(n_rating, float(n_rating)/m/n))

    rating_subs = np.where(R != 99)
    assert(len(rating_subs[0]) == n_rating)

    n_train = int(n_rating * p_train)
    n_valid = int(n_rating * (1-p_train))

    print('splitting with train: {}, test: {}'.format(n_train,
                                                                 n_valid))

    train_inds = sorted(np.random.choice(range(n_rating), n_train, False))
    test_inds = sorted(list(set(range(n_rating)) - set(train_inds)))

    R_train = scipy.sparse.csr_matrix((m, n))
    R_test = scipy.sparse.csr_matrix((m, n))

    for R_t, inds_t in zip((R_train, R_test), (train_inds, test_inds)):
        data = R[rating_subs[0][inds_t], rating_subs[1][inds_t]]
        spm = scipy.sparse.csr_matrix((data, (rating_subs[0][inds_t], rating_subs[1][inds_t])), shape=(m, n))
        R_t[:] = spm

    with open('./data/train_test.p', 'wb') as f:
        pickle.dump([R_train, R_test], f)

    return R_train, R_test

if __name__ == '__main__':
    load_train_test()