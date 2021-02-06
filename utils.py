import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay


class IsotropicMesh:
    def __init__(self, nodes, infimum=1e-8):
        """
        +--------------+------------------+-------+
        | Tensor       | shape            | type  |
        +--------------+------------------+-------+
        | nodes        | [NN, ND]         | float |
        | mask         | [NN]             | bool  |
        | simplices    | [NT, ND+1]       | int   |
        | surfaces     | [NT, ND+1, ND]   | int   |
        | tensor       | [NT, ND+1, ND+1] | float |
        | minors       | [NT, ND+1, ND+1] | float |
        | determinants | [NT]             | float |
        +--------------+------------------+-------+
        """
        self.nodes, (self.nn, self.dim) = nodes, nodes.shape

        # generate anisotropic simplices
        delaunay = Delaunay(self.nodes)
        simplices = delaunay.simplices
        volumes = np.linalg.det(self.nodes[simplices[:, :-1]] - self.nodes[simplices[:, [-1]]])
        valid_indices = [i for i, v in enumerate(volumes) if np.abs(v) > infimum]
        simplices = simplices[valid_indices, :]
        volumes = volumes[valid_indices]

        # generate mask of convex hull
        mask = np.expand_dims(self.nodes[delaunay.convex_hull], axis=1) - np.expand_dims(self.nodes, axis=(0, 2))
        self.mask = np.min(np.abs(np.linalg.det(mask)), axis=0) == 0

        # generate isotropic simplices
        def reverse(spx): return spx[[0, 2, 1] + list(range(3, spx.__len__()))]
        self.simplices = np.array([spx if flag else reverse(spx) for spx, flag in zip(simplices, volumes > 0)])
        self.nt = self.simplices.shape[0]

        # generate isotropic surfaces of simplices.
        self.surfaces = np.stack([np.roll(self.simplices, -k, 1) if k * self.dim % 2 else reverse(np.roll(self.simplices, -k, 1).T).T for k in range(self.dim + 1)], axis=1)[:, :, 1:]

        # generate minors and determinants of `tensor` in isotropic mode.
        #                   +- x_{t,0,0}  & \cdots & x_{t,0,ND-1}  & 1      -+
        # tensor[t, :, :] = |  \vdots     & \ddots & \vdots        & \vdots  |
        #                   +- x_{t,ND,0} & \cdots & x_{t,ND,ND-1} & 1      -+
        tensor = np.concatenate([self.nodes[self.simplices, :], np.ones(shape=(self.simplices.__len__(), self.dim + 1, 1))], axis=-1)
        self.determinants = np.abs(volumes)
        self.minors = np.einsum("tij,t->tji", np.linalg.inv(tensor), self.determinants)

    @classmethod
    def factorial(cls, k): return 1 if k <= 1 else k * cls.factorial(k - 1)

    def matrix(self, item_1, item_2, weights_1=None, weights_2=None):
        """
        +---------+---------+------------------+--------------+-------+
        | item_1  | item_2  | tensor shape     | matrix shape | type  |
        +---------+---------+------------------+--------------+-------+
        | p0      | p0      | [NT]             | [NT]         | float |
        | p0      | p1      | [NT, ND+1]       | [NT, NN]     | float |
        | p0      | grad_p1 | [NT, ND+1]       | [NT, NN]     | float |
        | p1      | p1      | [NT, ND+1, ND+1] | [NN, NN]     | float |
        | p1      | grad_p1 | [NT, ND+1, ND+1] | [NN, NN]     | float |
        | grad_p1 | grad_p1 | [NT, ND+1, ND+1] | [NN, NN]     | float |
        +---------+---------+------------------+--------------+-------+
        """
        if item_1 == "p0" and item_2 == "p0":
            tensor = 1 / self.factorial(self.dim) * self.determinants
        elif item_1 == "p0" and item_2 == "p1":
            tensor = 1 / self.factorial(self.dim + 1) * np.einsum("t,v->tv", self.determinants, np.ones(self.dim + 1))
        elif item_1 == "p0" and item_2 == "grad_p1":
            tensor = 1 / self.factorial(self.dim) * np.einsum("tvd,d->tv", self.minors[:, :, :-1], weights_2)
        elif item_1 == "p1" and item_2 == "p1":
            tensor = 1 / self.factorial(self.dim + 2) * np.einsum("t,ij->tij", self.determinants, np.ones((self.dim + 1, self.dim + 1)) + np.eye(self.dim + 1))
        elif item_1 == "p1" and item_2 == "grad_p1":
            tensor = 1 / self.factorial(self.dim + 1) * np.einsum("i,tjd,d->tij", np.ones(self.dim + 1), self.minors[:, :, :-1], weights_2)
        elif item_1 == "grad_p1" and item_2 == "grad_p1":
            tensor = 1 / self.factorial(self.dim) * np.einsum("tid,tjd,t->tij", self.minors[:, :, :-1], self.minors[:, :, :-1], 1 / self.determinants)
        else:
            raise ValueError("`{}` or `{}` is invalid. You should add them in this method.")

        if item_1 in ["p0"] and item_2 in ["p0"]:  # convert `simplex` X `simplex` tensor to matrix
            return np.diag(tensor)
        if item_1 in ["p0"] and item_2 in ["p1", "grad_p1"]:  # convert `simplex` X `node` tensor to matrix
            data = np.hstack([tensor[:, i] for i in range(self.dim + 1)])
            row = np.hstack([np.arange(self.nt) for _ in range(self.dim + 1)])
            col = np.hstack([self.simplices[:, i] for i in range(self.dim + 1)])
            return coo_matrix((data, (row, col)), shape=(self.nt, self.nn)).toarray()
        if item_1 in ["p1", "grad_p1"] and item_2 in ["p1", "grad_p1"]:  # convert `node` X `node` tensor to matrix
            data = np.hstack([tensor[:, i, j] for i in range(self.dim + 1) for j in range(self.dim + 1)])
            row = np.hstack([self.simplices[:, i] for i in range(self.dim + 1) for _ in range(self.dim + 1)])
            col = np.hstack([self.simplices[:, j] for _ in range(self.dim + 1) for j in range(self.dim + 1)])
            return coo_matrix((data, (row, col)), shape=(self.nn, self.nn)).toarray()

    def rhs(self, func, item):
        """
        +---------+--------------+--------------+-------+
        | item_2  | tensor shape | matrix shape | type  |
        +---------+--------------+--------------+-------+
        | p0      | [NT]         | [NT, 1]      | float |
        | p1      | [NT, ND+1]   | [NN, 1]      | float |
        +---------+--------------+--------------+-------+
        * `func` must be a scalar function.
        """
        f_val = func(np.mean(self.nodes[self.simplices, :], axis=1))  # `func` must be a scalar function.
        volumes = (1 / self.factorial(self.dim) * self.determinants)
        if item == "p0":
            tensor = f_val * volumes
        elif item == "p1":
            tensor = np.einsum("t,v->tv", f_val * (1 / (self.dim + 1)) * volumes, np.ones(shape=(self.dim + 1, )))
        else:
            raise ValueError("`{}` or `{}` is invalid. You should add them in this method.")

        if item in ["p0"]:  # convert `simplex` tensor to matrix
            return np.reshape(tensor, (-1, 1))
        elif item in ["p1"]:  # convert `node` tensor to matrix
            data = np.hstack([tensor[:, i] for i in range(self.dim + 1)])
            row = np.hstack([self.simplices[:, i] for i in range(self.dim + 1)])
            col = np.zeros_like(row)
            return coo_matrix((data, (row, col)), shape=(self.nn, 1)).toarray()

    def error(self, func_u, u_h, item, order=2):
        exact_u = func_u(np.mean(self.nodes[self.simplices, :], axis=1))
        volumes = (1 / self.factorial(self.dim) * self.determinants)
        def norm(x, y): return np.power(np.sum(np.power(np.abs(x - y), order) * volumes), 1 / order)
        if item == "p0":
            return norm(exact_u, u_h)
        if item == "p1":
            return norm(exact_u, np.mean(u_h[self.simplices], axis=1))


if __name__ == "__main__":
    def estimate_integer(func, points, num_refine=0):
        # convert to vector function
        is_scalar = func(points).shape.__len__() == 1
        if is_scalar:
            def vec_func(x): return np.reshape(func(x), (-1, 1))
        else:
            vec_func = func

        # refine current simplex
        dim = points.shape[-1]
        while num_refine > 0:
            nn = points.shape[0]
            edges = Delaunay(points).simplices[:, [[i, j] for i in range(1, dim + 1) for j in range(i)]]  # [NT, NE', 2]
            indices = [[i // nn, i % nn] for i in set(np.reshape(edges[:, :, 0] * nn + edges[:, :, 1], (-1,)))]
            points = np.vstack([points, np.mean(points[indices, :], axis=1)])  # [NN, ND]
            num_refine -= 1

        # compute all integers in fine simplices.
        simplices = Delaunay(points).simplices  # [NT, ND+1]
        tensor = vec_func(points[simplices.flatten(), :])  # [NT*(ND+1), NF]
        tensor = np.reshape(tensor, newshape=(simplices.shape[0], simplices.shape[1], -1))  # [NT, ND+1, NF]
        volumes = np.abs(np.linalg.det(points[simplices[:, :-1], :] - points[simplices[:, [-1]], :]))  # [NT]
        tensor = 1 / IsotropicMesh.factorial(dim) * np.einsum("tvd,t->vd", tensor, volumes)

        return np.mean(tensor) if is_scalar else np.mean(tensor, axis=0)

    # check method `estimate_integer`:
    #                                                 \Pi_i \alpha_i!
    # \int_K \Pi_i \lambda_i ^ {\alpha_i}  dxdy = ------------------------ * determinant
    #                                             (dim + \Sum_i \alpha_i)!
    for ix in range(3):
        for jx in range(3):
            if ix + jx > 2:
                continue
            result = estimate_integer(
                func=lambda x: x[:, 0] ** ix * x[:, 1] ** jx,
                points=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]),
                num_refine=2 + ix + jx
            )
            print(ix, jx, result)
            assert np.abs(result - IsotropicMesh.factorial(ix) * IsotropicMesh.factorial(jx) / IsotropicMesh.factorial(3 + ix + jx)) / result < 0.01

    # Check `tensor`, `minors`, and `determinants`.
    mesh = IsotropicMesh(nodes=np.array([[ix // 4, ix % 4 // 2, ix % 2] for ix in range(8)], dtype=np.float))
    assert np.min(mesh.determinants) > 1e-8
    vectors = mesh.nodes[mesh.simplices] - np.mean(mesh.nodes[mesh.surfaces], axis=2)
    normals = -mesh.minors[:, :, :-1]  # outer-pointing normal
    assert np.linalg.norm(np.einsum("svd,svd->sv", vectors, normals)[:, 1] + mesh.determinants) < 1e-6
    p1 = np.einsum("tvd,tvd->tv", mesh.minors[:, :, :-1], mesh.nodes[mesh.simplices, :]) + mesh.minors[:, :, -1]
    assert np.linalg.norm(p1[:, np.random.randint(mesh.dim)] - mesh.determinants) < 1e-6
