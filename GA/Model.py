import numpy as np


def k_tournament_generator(n, k):
    """Create 2*n k-tournaments with sample size n."""
    a = np.vstack([np.random.choice(np.arange(n), size=2 * k, replace=False) for _ in range(n)])
    b = a[:, k:]
    a = a[:, :k]
    T = np.zeros((2 * n, k), dtype=np.int)
    T[0::2, :] = a
    T[1::2, :] = b
    return T


class PartitionModel(object):
    def __init__(self, m, n, fitness, bounds=None, beq=None, k=2, gamma=0.0, lmbda=0.0, gain=1):
        self._fitness = fitness
        self._bounds = bounds
        self._beq = beq
        if self._bounds is None:
            self._bounds = (0, n-1)
        self._k = k
        self._gamma = gamma
        self._lambda = lmbda
        self._gain = gain
        self._chromosomes = np.random.randint(self._bounds[0], self._bounds[1] + 1, size=(m, n))
        self._f, self._g = self._fitness(self._chromosomes)

    def update(self):
        # Do Select
        p = self._select()
        # Do Crossover
        c = self._xover(p)
        # Do Mutation
        c = self._mutate(c)
        # Do Survival
        fc, gc = self._fitness(c)
        self._f = np.hstack((self._f, fc))
        self._g = np.hstack((self._g, gc))
        idx = self._survival()
        self._chromosomes = np.vstack((self._chromosomes, c))[idx, :]
        self._f = self._f[idx]
        self._g = self._g[idx]
        #print(self._chromosomes.std(axis=0))
        return self._chromosomes[0, :], self._f[0], self._g[0]

    def _select(self):
        t_idx = k_tournament_generator(self._chromosomes.shape[0], self._k)
        f = self._gain*self._g + self._f
        t = f[t_idx]
        idx = np.argmin(t, axis=1)
        p_idx = t_idx[range(2*self._chromosomes.shape[0]),idx]
        return self._chromosomes[p_idx, :]

    def _xover(self, p):
        I = np.arange(self._chromosomes.shape[1])
        p1 = p[::2, :]
        p2 = p[1::2, :]
        idx = np.random.choice(I, (2,), False)
        c = p1
        p = np.random.random((c.shape[0],)) < self._gamma
        if np.any(p):
            c[np.ix_(p, idx)] = p2[np.ix_(p, idx)]
        return c

    def _mutate(self, c):
        p = np.random.random((c.shape[0],)) < self._lambda
        m = c[p, :]
        o = np.random.randint(self._bounds[0], self._bounds[1]+1, (m.shape[0], 1))
        i = np.random.randint(m.shape[1], size=(m.shape[0],))
        m[:, i] = o
        c[p, :] = m
        return c

    def _survival(self):
        f = self._f + self._gain*self._g
        idx = np.argsort(f)
        idx = np.hstack((idx[:int(self._chromosomes.shape[0]/2)],
                         idx[self._chromosomes.shape[0]+1:self._chromosomes.shape[0]+int(self._chromosomes.shape[0]/2)+1]))
        return idx
        #return idx[:int(self._chromosomes.shape[0])]
