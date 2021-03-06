import numpy as np
import torch
import numpy.linalg as LA


class TuckerALS():
    """Methods for Tucker-ALS (alternating least squares) tensor decomposition.

    Parameters
    ----------
    max_iter : integer (default: 300)
        Maximum number of ALS iterations.
    tol : scalar (default: 1e-6)
        Tolerance for ALS in terms of change in relative error.
    """
    def __init__(self, max_iter=300, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.err = [1]
        self.delta_err = np.inf
        self.itr = 0

    def _order(self, n):
        return np.delete(np.arange(self.N), n)

    def _signflip(self, X):
        idx = np.argmax(np.abs(X), axis=0)
        signs = np.sign(X[idx, np.arange(X.shape[1])])
        return X * signs

    def _tenmat(self, D, n, as_ten=False):
        dims = np.array(D.shape)
        M = dims[n]
        N = np.prod(dims[self._order(n)])
        permute = np.r_[n, self._order(n)]
        D_ = D.transpose(permute).reshape((M,N))
        if as_ten:
            return torch.as_tensor(D_, dtype=torch.double)
        else:
            return D_

    def _tenmat_prod(self, X, A_, rank, n):
        Y = A_.T @ self._tenmat(X, n)
        I = np.array(X.shape)
        Y = Y.reshape(np.r_[rank[n], I[self._order(n)]])
        permute = np.argsort(np.r_[n, self._order(n)])
        return Y.transpose(permute)

    def _hosvd_init(self, rank):
        A_ = []
        for n in range(self.N):
            X_ = self._tenmat(self.A, n, as_ten=True)
            A_ += [
                self._signflip(np.array(torch.svd(X_)[0])[:,:rank[n]])
            ]
        return A_

    def _compute_Y(self, A_, rank, n):
        modes = self._order(n)
        Y = self._tenmat_prod(self.A, A_[modes[0]], rank, modes[0])
        for i in modes[1:]:
            Y = self._tenmat_prod(Y, A_[i], rank, i)
        return Y

    def _tensor_svd(self, Y, rank, n):
        A_ = self._tenmat(Y, n, as_ten=True)
        A_ = np.array(torch.svd(A_)[0])[:,:rank[n]]
        if A_.shape[1] < rank[n]:
            col = np.zeros((A_.shape[0], rank[n] - A_.shape[1]))
            return self._signflip(np.c_[A_, col])
        else:
            return self._signflip(A_)

    def _compute_core(self, A_, rank):
        G = self._tenmat_prod(self.A, A_[0], rank, 0)
        for n in np.arange(1, len(A_)):
            G = self._tenmat_prod(G, A_[n], rank, n)
        return G

    def _reconstruct_tensor(self, G, A_):
        B = np.einsum('ijk,ai->ajk', G, A_[0])
        B = np.einsum('ajk,bj->abk', B, A_[1])
        B = np.einsum('abk,ck->abc', B, A_[2])
        return B

    def _convert_err(self, A, err):
        """Converts relative error returned by the 'als' to SSE
        """
        return np.square(err * LA.norm(A))

    def _als_output(self):
        print(f'Number of iterations: {self.itr}')
        print('Relative reconstruction error: {:.4f}'.format(self.err))
        return None

    def _rs_output(self, label, aic, opt):
        print(f'Label: {label}')
        print('Minimum AIC: {:.2f}'.format(np.min(aic)))
        print(f'Optimal Ranks: {opt}')
        return None

    def als(self, A, rank, verbose=0):
        """Third-order rank decomposition of tensor A using Tucker ALS.

        Parameters
        ----------
        self : object
        A : array-like tensor, shape = (I, J, K)
            The third-order tensor to be decomposed.
        rank : array-like, length = 3
            Desired rank-(I, J, K) of the decomposed core tensor.
        verbose : boolean

        Returns
        -------
        self : object
        """
        self.A = A
        self.N = A.ndim
        A_ = self._hosvd_init(rank)
        while self.delta_err >= self.tol and self.itr < self.max_iter:
            for n in range(self.N):
                Y = self._compute_Y(A_, rank, n)
                A_[n] = self._tensor_svd(Y, rank, n)
            G = self._compute_core(A_, rank)
            B = self._reconstruct_tensor(G, A_)
            self.err += [LA.norm(self.A - B) / LA.norm(self.A)]
            self.delta_err = self.err[self.itr] - self.err[self.itr + 1]
            self.itr += 1
        self.B, self.G, self.A_ = B, G, A_
        self.err = self.err[-1]
        if verbose:
            self._als_output()
        return self

    def rank_selection(self, A, ranks, label='', verbose=1):
        """Obtain the optimal low-rank representation of 'A' using Akaike
        information criterion (AIC).

        Parameters
        ----------
        A : array-like tensor, shape = (I, J, K)
            The third-order tensor to be decomposed.
        ranks : list of array-likes, length = 3
            Desired ranks to iterate over for each dimension.
        label : string or integer
        verbose : boolean

        Returns
        -------
        self : object
        """
        R1, R2, R3 = ranks
        aic = np.zeros((len(R1), len(R2), len(R3)))
        for i, I in enumerate(R1):
            for j, J in enumerate(R2):
                for k, K in enumerate(R3):
                    tucker = self.als(A, [I, J, K])
                    sse = self._convert_err(tucker.A, tucker.err)
                    aic[i,j,k] = 2*sse + 2*(I + J + K)
                    self.__init__()
        idx = np.array(np.where(aic == np.min(aic))).flatten()
        opt = np.array([R1[idx[0]], R2[idx[1]], R3[idx[2]]])
        if verbose:
            self._rs_output(label, aic, opt)
        return self.als(A, opt)
