from enum import Enum
from typing import List, Tuple
from qecc.StabilizerCode import StabilizerCode, vectorized_mod2

import numpy as np
import networkx as nx


class EdgeType(Enum):
    H = 0
    V = 1


class Edge:
    def __init__(self, x: int, y: int, t: EdgeType):
        self.x = x
        self.y = y
        self.t = t

    def __repr__(self):
        return '(' + str(self.x) + ',' + str(self.y) + ',' + ('H' if self.t == EdgeType.H else 'V') + ')'

    def dual(self):
        """
        Return the dual edge.
        """
        if self.t == EdgeType.V:
            return Edge(self.x + 1, self.y, EdgeType.H)
        else:
            return Edge(self.x, self.y + 1, EdgeType.V)


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __repr__(self):
        return '(' + str(self.x) + ',' + str(self.y) + ')'

    def plaquette(self) -> List[Edge]:
        """
        Return a list of edges that is the plaquette, whose upper left vertex is this point.
        """
        return [Edge(self.x, self.y, EdgeType.H),
                Edge(self.x, self.y, EdgeType.V),
                Edge(self.x + 1, self.y, EdgeType.H),
                Edge(self.x, self.y + 1, EdgeType.V)]

    def site(self) -> List[Edge]:
        """
        Return a list of edges with this point as one of its vertex.
        """
        return [Edge(self.x, self.y, EdgeType.H),
                Edge(self.x, self.y, EdgeType.V),
                Edge(self.x - 1, self.y, EdgeType.V),
                Edge(self.x, self.y - 1, EdgeType.H)]


class ToricCode(StabilizerCode):
    def __init__(self, x: int, y: int):
        """
        Initialize the toric code as a stabilizer code.

        Args:
            x: The length of the torus lattice on x direction.
            y: The length of the torus lattice on y direction.
        """
        self._x = x
        self._y = y
        stabilizers_str, logic_str = self.__generate_stabilizers()
        super().__init__(stabilizers_str, logic_str)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __generate_stabilizers(self) -> Tuple[List[str], List[str]]:
        """
        Generate the string list of stabilizers and logic operators for toric code.

        Returns:
            (stabilizer_str, logic_str): String list for stabilizers and logic operators.
        """
        stabilizer_str, logic_str = [], []

        for i in range(self.x):
            for j in range(self.y):
                pt = Point(i, j)

                string = list('I' * 2 * self.x * self.y)
                for e in pt.site():
                    string[self.__to_idx(e)] = 'X'
                stabilizer_str.append(''.join(string))

                string = list('I' * 2 * self.x * self.y)
                for e in pt.plaquette():
                    string[self.__to_idx(e)] = 'Z'
                stabilizer_str.append(''.join(string))
        # pop last two X and Z stabilizers since all stabilizers of the same kind multiply to identity
        stabilizer_str.pop()
        stabilizer_str.pop()

        # the order of the logical operators must be maintained
        string = list('I' * 2 * self.x * self.y)
        for j in range(self.y):
            string[self.__to_idx(Edge(0, j, EdgeType.V))] = 'X'
        logic_str.append(''.join(string))

        string = list('I' * 2 * self.x * self.y)
        for i in range(self.x):
            string[self.__to_idx(Edge(i, 0, EdgeType.V))] = 'Z'
        logic_str.append(''.join(string))

        string = list('I' * 2 * self.x * self.y)
        for i in range(self.x):
            string[self.__to_idx(Edge(i, 0, EdgeType.H))] = 'X'
        logic_str.append(''.join(string))

        string = list('I' * 2 * self.x * self.y)
        for j in range(self.y):
            string[self.__to_idx(Edge(0, j, EdgeType.H))] = 'Z'
        logic_str.append(''.join(string))

        return stabilizer_str, logic_str

    def __to_idx(self, obj) -> int:
        """
        Convert a point ot an edge to its corresponding index. Row first. For edge, horizontal edges always come before
        vertical edges.

        Args:
            obj: A point or an edge.

        Reutns:
            The corresponding index.
        """
        if isinstance(obj, Edge):
            return (obj.x % self.x) * self.y + (obj.y % self.y) + (self.x * self.y if obj.t == EdgeType.V else 0)
        elif isinstance(obj, Point):
            return (obj.x % self.x) * self.y + (obj.y % self.y)

    def __to_point(self, idx: int) -> Point:
        """
        Convert the index to a point.

        Args:
            idx: Point index.

        Returns:
            A point.
        """
        y = idx % self.y
        x = (idx - y) // self.x
        return Point(x, y)

    def __distance(self, pt1: Point, pt2: Point) -> int:
        """
        Return the length of the shortest path connecting two points on the torus lattice.

        Args:
            pt1: Point 1.
            pt2: Point 2.

        Returns:
            The shortest path length.
        """

        return min(abs(pt1.x - pt2.x), self.x - abs(pt1.x - pt2.x)) + \
               min(abs(pt1.y - pt2.y), self.y - abs(pt1.y - pt2.y))

    def __shortest_path(self, pt1: Point, pt2: Point) -> List[Edge]:
        """
        Return the shortest path connecting two points on the torus lattice. The path is composed of a list of edges.

        Args:
            pt1: Point 1.
            pt2: Point 2.

        Returns:
            A list of edges which is the shortest path.
        """
        def sign(x: int):
            return 1 if x >= 0 else -1

        p1, p2 = pt1, pt2
        x_dir = sign(p2.x - p1.x)
        if abs(pt1.x - pt2.x) > self.x - abs(pt1.x - pt2.x):
            p2.x -= x_dir * self.x
            x_dir = -x_dir
        x_path = [Edge((x - (0 if x_dir > 0 else 1)) % self.x, p1.y, EdgeType.V) for x in range(p1.x, p2.x, x_dir)]
        y_dir = sign(p2.y - p1.y)
        if abs(pt1.y - pt2.y) > self.y - abs(pt1.y - pt2.y):
            p2.y -= y_dir * self.y
            y_dir = -y_dir
        y_path = [Edge(p2.x % self.x, (y - (0 if y_dir > 0 else 1)) % self.y, EdgeType.H) for y in range(p1.y, p2.y, y_dir)]

        assert self.__distance(pt1, pt2) == len(x_path) + len(y_path)
        return x_path + y_path

    def mwm_correction(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Eliminate the syndrome through minimal weight perfect matching.

        Args:
            syndrome: A (n - k) x 1 array of syndromes.

        Returns:
            A 2n x 1 array of physical operators.
        """
        assert len(syndrome) == self.n - self.k

        # synd_x is the syndrome caused by X errors measured by Z stabilizers
        synd_x, synd_z = np.zeros([self.x, self.y], dtype=np.int), np.zeros([self.x, self.y], dtype=np.int)

        # locate syndrome in the lattice
        idx = 0
        for i in range(self.x):
            for j in range(self.y):
                if i == self.x - 1 and j == self.y - 1:
                    break
                synd_z[i, j] = syndrome[idx]
                idx += 1
                synd_x[i, j] = syndrome[idx]
                idx += 1

        # infer the redundant last syndrome through parity
        if synd_z.sum() % 2 == 1:
            synd_z[self.x - 1, self.y - 1] = 1
        if synd_x.sum() % 2 == 1:
            synd_x[self.x - 1, self.y - 1] = 1

        synd_x, synd_z = vectorized_mod2(synd_x), vectorized_mod2(synd_z)

        # construct graph
        G_x, G_z = nx.Graph(), nx.Graph()
        for i in range(self.x):
            for j in range(self.y):
                p1 = Point(i, j)
                if synd_x[i, j] == 1:
                    for a in range(self.x):
                        for b in range(self.y):
                            if synd_x[a, b] == 1 and (not (i == a and j == b)):
                                p2 = Point(a, b)
                                G_x.add_edge(self.__to_idx(p1), self.__to_idx(p2), weight=-self.__distance(p1, p2))
                if synd_z[i, j] == 1:
                    for a in range(self.x):
                        for b in range(self.y):
                            if synd_z[a, b] == 1 and (not (i == a and j == b)):
                                p2 = Point(a, b)
                                G_z.add_edge(self.__to_idx(p1), self.__to_idx(p2), weight=-self.__distance(p1, p2))

        # minimal weight perfect matching
        phys_err = np.zeros([2 * self.n, 1], dtype=np.int)
        for p in nx.max_weight_matching(G_z, maxcardinality=True):
            paths_z = self.__shortest_path(self.__to_point(p[0]), self.__to_point(p[1]))
            for e in paths_z:
                phys_err[self.__to_idx(e) + self.n] += 1
        for p in nx.max_weight_matching(G_x, maxcardinality=True):
            paths_x = self.__shortest_path(self.__to_point(p[0]), self.__to_point(p[1]))
            for e in paths_x:
                phys_err[self.__to_idx(e.dual())] += 1

        return phys_err
