import numpy as np

from static.algorithms.simple_FEM.utils import IsotropicMesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def numerical_gradients(func, x, eps=1e-6):
    dim = x.shape[-1]
    gradients = []
    for d in range(dim):
        vec = np.eye(dim)[[d]]
        gradients.append((func(x + eps * vec) - func(x - eps * vec)) / (2 * eps))
    return np.stack(gradients, axis=1)


def numerical_laplace(func, x, eps=1e-6):
    dim = x.shape[-1]
    second_derivative = 0
    for d in range(dim):
        vec = np.eye(dim)[[d]]
        second_derivative = second_derivative + (func(x + eps * vec) - 2 * func(x) + func(x - eps * vec)) / eps ** 2
    return second_derivative


def cube_region(num=3):
    from scipy.spatial import Delaunay

    def refine_mesh(points):
        nn, dim = points.shape
        edges = Delaunay(points).simplices[:, [[i, j] for i in range(1, dim + 1) for j in range(i)]]  # [NT, NE, 2]
        row = np.minimum(edges[:, :, 0].flatten(), edges[:, :, 1].flatten())
        col = np.maximum(edges[:, :, 0].flatten(), edges[:, :, 1].flatten())
        indices = np.unique(np.stack([row, col], axis=1), axis=0)
        points = np.vstack([points, np.mean(points[indices, :], axis=1)])
        return np.unique(points, axis=0)

    nodes = np.vstack([-1 + 2 * np.array([[i // 4, i % 4 // 2, i % 2] for i in range(8)], dtype=np.float), np.array([[0., 0., 0.]])])
    for _ in range(num):
        nodes = refine_mesh(nodes)
    return nodes


def cassini_oval_region(num=16):
    """(x^2 + y^2)^2 - 2(x^2 - y^2) = 3"""
    from skimage import measure

    # generate dense points in bound
    phi = np.linspace(0, 2 * np.pi, num=10000, endpoint=False)
    r = np.sqrt(np.cos(2 * phi) + np.sqrt(4 - np.sin(2 * phi) ** 2))
    dense_boundary_points = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)

    # generate nodes in bound
    phi = np.linspace(0, 2 * np.pi, num=num, endpoint=False)
    r = np.sqrt(np.cos(2 * phi) + np.sqrt(4 - np.sin(2 * phi) ** 2))
    boundary_points = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)

    # generate nodes in interal region
    gap = np.min(np.linalg.norm(boundary_points[:-1] - boundary_points[1:], axis=1))
    xv, yv = np.meshgrid(np.arange(-2, 2, gap), np.arange(-1.5, 1.5, gap))
    unsafe_inner_points = np.stack([xv.flatten(), yv.flatten()], axis=1)
    masks = measure.points_in_poly(points=unsafe_inner_points, verts=boundary_points)
    unsafe_inner_points = unsafe_inner_points[[i for i, mask in enumerate(masks) if mask]]
    distance = np.linalg.norm(np.expand_dims(unsafe_inner_points, axis=1) - np.expand_dims(dense_boundary_points, axis=0), axis=-1)
    inner_points = unsafe_inner_points[[i for i, dis in enumerate(np.min(distance, axis=1)) if dis > 0.5 * gap]]

    return np.vstack([boundary_points, inner_points])


def example_cube3d(path=None, num_refine=3):
    """
    Problem:
        -\Delta u + \boldsymbol{c} \cdot \nabla u + u = f, \quad \boldsymbol{x} \in \Omega
        u = g, \quad \boldsymbol{x} \in \partial \Omega

    Combination of basis functions:
        u_h = \sum_i (p_i \psi_i), \quad \psi_i \in P_1(T_h)

    The weak form:
        \sum_j ((\nabla \psi_i, \nabla \psi_j) + (\psi_i, \boldsymbol{c} \cdot \nabla \psi_j) + (\psi_i, \psi_j)) p_i = (\psi_i, f)
    """
    dim = 3
    def func_u(x): return np.exp(-np.sum(np.square(x), axis=1)) * np.sum(np.cos(x), axis=1)
    def func_f(x): return np.exp(-np.sum(np.square(x), axis=1)) * ((2 * dim + 2 - 4 * np.sum(np.square(x), axis=1) - 2 * np.sum(x, axis=1)) * np.sum(np.cos(x), axis=1) - 4 * np.sum(x * np.sin(x), axis=1) - np.sum(np.sin(x), axis=1))
    def func_g(x): return func_u(x)

    # check PDEs
    def estimate_func_f(x): return -numerical_laplace(func_u, x) + np.sum(numerical_gradients(func_u, x), axis=1) + func_u(x)
    check_point = np.random.rand(1, dim)
    print("check:", func_f(check_point), estimate_func_f(check_point))

    # Start to solve
    mesh = IsotropicMesh(nodes=cube_region(num_refine))
    boundary_indices = [i for i, mask in enumerate(mesh.mask) if mask]
    inner_indices = [i for i, mask in enumerate(mesh.mask) if not mask]

    matrix = np.zeros(shape=(mesh.nn, mesh.nn), dtype=np.float)
    rhs = np.zeros(shape=(mesh.nn, 1), dtype=np.float)
    matrix[boundary_indices, boundary_indices] += 1
    matrix[inner_indices, :] += mesh.matrix(item_1="grad_p1", item_2="grad_p1")[inner_indices, :]
    matrix[inner_indices, :] += mesh.matrix(item_1="p1", item_2="grad_p1", weights_2=np.ones(dim))[inner_indices, :]
    matrix[inner_indices, :] += mesh.matrix(item_1="p1", item_2="p1")[inner_indices, :]
    rhs[boundary_indices, :] = np.reshape(func_g(mesh.nodes[boundary_indices, :]), (-1, 1))
    rhs[inner_indices, :] = np.reshape(mesh.rhs(func=func_f, item="p1")[inner_indices], newshape=(-1, 1))

    coeff = np.linalg.solve(matrix, rhs.flatten())

    distant = np.min(np.linalg.norm(np.expand_dims(mesh.nodes[inner_indices, :], axis=1) - np.expand_dims(mesh.nodes[boundary_indices, :], axis=0), axis=-1), axis=1)
    hull_indices = [inner_indices[i] for i, dis in enumerate(distant) if dis == np.min(distant)]
    hull_vertices = mesh.nodes[hull_indices, :]
    from scipy.spatial import Delaunay
    convex_hull = Delaunay(hull_vertices).convex_hull
    exact_u = func_u(np.mean(hull_vertices[convex_hull, :], axis=1))
    numerical_u = np.mean(coeff[hull_indices][convex_hull], axis=1)
    if path is None:
        return {
            "exact_colors": np.array([[u, 0, 1 - u] for u in (exact_u - np.min(exact_u)) / (np.max(exact_u) - np.min(exact_u))]),
            "numerical_colors": np.array([[u, ] * 3 for u in (numerical_u - np.min(numerical_u)) / (np.max(numerical_u) - np.min(numerical_u))]),
            "polygons": hull_vertices[convex_hull, :],
            "l2": mesh.error(func_u, coeff, item="p1")
        }

    # Show figure
    fig = plt.figure(figsize=(10, 6))

    exact_ax = fig.add_subplot(1, 2, 1, projection='3d')
    colors = np.array([[u, 0, 1 - u] for u in (exact_u - np.min(exact_u)) / (np.max(exact_u) - np.min(exact_u))])
    exact_ax.add_collection(Poly3DCollection(hull_vertices[convex_hull, :], color=colors, alpha=0.5, linewidth=0))
    exact_ax.set_xlim(-1, 1)
    exact_ax.set_ylim(-1, 1)
    exact_ax.set_zlim(-1, 1)
    exact_ax.set_xlabel("x")
    exact_ax.set_ylabel("y")
    exact_ax.set_zlabel("z")
    exact_ax.set_title("exact u")

    numerical_ax = fig.add_subplot(1, 2, 2, projection='3d')
    colors = np.array([[u, 0, 1 - u] for u in (numerical_u - np.min(numerical_u)) / (np.max(numerical_u) - np.min(numerical_u))])
    numerical_ax.add_collection(Poly3DCollection(hull_vertices[convex_hull, :], color=colors, alpha=0.5, linewidth=0))
    numerical_ax.set_xlim(-1, 1)
    numerical_ax.set_ylim(-1, 1)
    numerical_ax.set_zlim(-1, 1)
    numerical_ax.set_xlabel("x")
    numerical_ax.set_ylabel("y")
    numerical_ax.set_zlabel("z")
    numerical_ax.set_title("numerical u(l2 error: {:.2e})".format(mesh.error(func_u, coeff, item="p1")))

    fig.savefig(path)


class Canvas3D:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 6))
        self.exact_ax = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.numerical_ax = self.fig.add_subplot(1, 2, 2, projection='3d')

        self.num_refine = 1
        self.draw(example_cube3d(path=None, num_refine=self.num_refine))

        self.fig.canvas.mpl_connect("key_press_event", self.key_press_event)
        plt.show()

    def draw(self, data):
        self.exact_ax.add_collection(Poly3DCollection(data["polygons"], color=data["exact_colors"], alpha=0.5, linewidth=0))
        self.exact_ax.set_xlim(-1, 1)
        self.exact_ax.set_ylim(-1, 1)
        self.exact_ax.set_zlim(-1, 1)
        self.exact_ax.set_xlabel("x")
        self.exact_ax.set_ylabel("y")
        self.exact_ax.set_zlabel("z")
        self.exact_ax.set_title("exact u")

        self.numerical_ax.add_collection(Poly3DCollection(data["polygons"], color=data["numerical_colors"], alpha=0.5, linewidth=0))
        self.numerical_ax.set_xlim(-1, 1)
        self.numerical_ax.set_ylim(-1, 1)
        self.numerical_ax.set_zlim(-1, 1)
        self.numerical_ax.set_xlabel("x")
        self.numerical_ax.set_ylabel("y")
        self.numerical_ax.set_zlabel("z")
        self.numerical_ax.set_title("numerical u(l2 error: {:.2e})".format(data["l2"]))

    def key_press_event(self, event):
        if event.key == 'up':
            self.num_refine = min(self.num_refine + 1, 6)
        elif event.key == 'down':
            self.num_refine = max(self.num_refine - 1, 1)
        else:
            return

        self.exact_ax.clear()
        self.numerical_ax.clear()
        self.draw(example_cube3d(path=None, num_refine=self.num_refine))
        self.fig.canvas.draw()


def example_mix2d(path=None, num_refine=3):
    """
    Problem:
        -\Delta u + v + u = f, \quad \boldsymbol{x} \in \Omega
        \boldsymbol{c} \cdot \nabla u - v = 0, \quad \boldsymbol{x} \in \Omega
        u = g, \quad \boldsymbol{x} \in \partial \Omega

    Combination of basis functions:
        u_h = \sum_i (p_i \psi_i^u), \quad \psi_i^u \in P_1(T_h)
        v_h = \sum_i (q_i \psi_i^v), \quad \psi_i^v \in P_0(T_h)

    The weak form:
        \sum_j ((\nabla \psi_i^u, \nabla \psi_j^u) + (\psi_i^u, \psi_j^u)) p_j + \sum_k (\psi_i^u, \psi_k^v) q_k = (\psi_i^u, f)
        \sum_j (\psi_i^v, \boldsymbol{c} \cdot \nabla \psi_j^u) p_j - \sum_k (\psi_i^v, \psi_k^v) q_k = 0
    """
    dim = 2
    def func_u(x): return np.exp(-np.sum(np.square(x), axis=1)) * np.sum(np.cos(x), axis=1)
    def func_v(x): return -np.exp(-np.sum(np.square(x), axis=1)) * (2 * np.sum(x, axis=1) * np.sum(np.cos(x), axis=1) + np.sum(np.sin(x), axis=1))

    def func_f(x): return np.exp(-np.sum(np.square(x), axis=1)) * ((5 - 4 * np.sum(np.square(x), axis=1)) * np.sum(np.cos(x), axis=1) - 4 * np.sum(x * np.sin(x), axis=1)) + func_v(x) + func_u(x)
    def func_g(x): return func_u(x)

    # check PDEs
    def estimate_func_f(x): return -numerical_laplace(func_u, x) + func_v(x) + func_u(x)
    def estimate_func_v(x): return np.sum(numerical_gradients(func_u, x), axis=1)
    check_point = np.random.rand(1, 2)
    print("check:", func_f(check_point), estimate_func_f(check_point), func_v(check_point), estimate_func_v(check_point))

    # Start to solve
    mesh = IsotropicMesh(nodes=cassini_oval_region(4 * 2 ** num_refine))
    boundary_indices = [i for i, mask in enumerate(mesh.mask) if mask]
    inner_indices = [i for i, mask in enumerate(mesh.mask) if not mask]

    matrix = np.zeros(shape=(mesh.nn + mesh.nt, mesh.nn + mesh.nt), dtype=np.float)
    rhs = np.zeros(shape=(mesh.nn + mesh.nt, 1), dtype=np.float)

    # u = g, \quad \boldsymbol{x} \in \partial \Omega
    matrix[boundary_indices, boundary_indices] += 1
    rhs[boundary_indices, :] = np.reshape(func_g(mesh.nodes[boundary_indices, :]), newshape=(-1, 1))

    # \sum_j ((\nabla \psi_i^u, \nabla \psi_j^u) + (\psi_i^u, \psi_j^u)) p_j + \sum_k (\psi_i^u, \psi_k^v) q_k = (\psi_i^u, f)
    matrix[inner_indices, :mesh.nn] = (mesh.matrix(item_1="grad_p1", item_2="grad_p1") + mesh.matrix(item_1="p1", item_2="p1"))[inner_indices, :]
    matrix[inner_indices, mesh.nn:] = mesh.matrix(item_1="p0", item_2="p1")[:, inner_indices].T
    rhs[inner_indices, :] = mesh.rhs(func=func_f, item="p1")[inner_indices, :]

    # \sum_j (\psi_i^v, \boldsymbol{c} \cdot \nabla \psi_j^u) p_j - \sum_k (\psi_i^v, \psi_k^v) q_k = 0
    matrix[mesh.nn:, :mesh.nn] = mesh.matrix(item_1="p0", item_2="grad_p1", weights_2=np.ones(dim))
    matrix[mesh.nn:, mesh.nn:] = -mesh.matrix(item_1="p0", item_2="p0")

    coeff = np.linalg.solve(matrix, rhs.flatten())
    exact_u, numerical_u = func_u(mesh.nodes), coeff[:mesh.nn]
    centers = np.mean(mesh.nodes[mesh.simplices], axis=1)
    exact_v, numerical_v = func_v(centers), coeff[mesh.nn:]
    if path is None:
        return {
            "l2_u": mesh.error(func_u, numerical_u, item="p1"),
            "nodes": mesh.nodes,
            "exact_u": exact_u,
            "numerical_u": numerical_u,
            "l2_v": mesh.error(func_v, numerical_v, item="p0"),
            "centers": centers,
            "exact_v": exact_v,
            "numerical_v": numerical_v,
        }

    # Show figure
    fig = plt.figure(figsize=(10, 6))

    ax_u = fig.add_subplot(1, 2, 1, projection='3d')
    ax_u.set_title("u: l2 error: {:.2e}".format(mesh.error(func_u, numerical_u, item="p1")))
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("y")
    ax_u.set_zlabel("u")
    ax_u.plot_trisurf(mesh.nodes[:, 0], mesh.nodes[:, 1], exact_u, alpha=0.5)
    ax_u.plot_trisurf(mesh.nodes[:, 0], mesh.nodes[:, 1], numerical_u, alpha=0.5)

    ax_v = fig.add_subplot(1, 2, 2, projection='3d')
    ax_v.set_title("v: l2 error: {:.2e}".format(mesh.error(func_v, numerical_v, item="p0")))
    ax_v.set_xlabel("x")
    ax_v.set_ylabel("y")
    ax_v.set_zlabel("v")
    ax_v.plot_trisurf(centers[:, 0], centers[:, 1], exact_v, alpha=0.5)
    ax_v.plot_trisurf(centers[:, 0], centers[:, 1], numerical_v, alpha=0.5)

    fig.savefig(path)


class Canvas2D:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 6))
        self.ax_u = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_v = self.fig.add_subplot(1, 2, 2, projection='3d')

        self.num_refine = 1
        self.draw(example_mix2d(path=None, num_refine=self.num_refine))

        self.fig.canvas.mpl_connect("key_press_event", self.key_press_event)
        plt.show()

    def draw(self, data):
        self.ax_u.set_title("u: l2 error: {:.2e}".format(data["l2_u"]))
        self.ax_u.set_xlabel("x")
        self.ax_u.set_ylabel("y")
        self.ax_u.set_zlabel("u")
        self.ax_u.plot_trisurf(data["nodes"][:, 0], data["nodes"][:, 1], data["exact_u"], alpha=0.5)
        self.ax_u.plot_trisurf(data["nodes"][:, 0], data["nodes"][:, 1], data["numerical_u"], alpha=0.5)

        self.ax_v.set_title("v: l2 error: {:.2e}".format(data["l2_v"]))
        self.ax_v.set_xlabel("x")
        self.ax_v.set_ylabel("y")
        self.ax_v.set_zlabel("v")
        self.ax_v.plot_trisurf(data["centers"][:, 0], data["centers"][:, 1], data["exact_v"], alpha=0.5)
        self.ax_v.plot_trisurf(data["centers"][:, 0], data["centers"][:, 1], data["numerical_v"], alpha=0.5)

    def key_press_event(self, event):
        if event.key == 'up':
            self.num_refine = min(self.num_refine + 1, 6)
        elif event.key == 'down':
            self.num_refine = max(self.num_refine - 1, 1)
        else:
            return

        self.ax_u.clear()
        self.ax_v.clear()
        self.draw(example_mix2d(path=None, num_refine=self.num_refine))
        self.fig.canvas.draw()


if __name__ == "__main__":
    example_cube3d(path="./cube3d.png", num_refine=2)
    example_mix2d(path="./mix2d.png", num_refine=2)
    plt.show()
