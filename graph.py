import cv2
import pygame.draw
from helper import *


class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.checked = False
        self.observed = False
        self.visited = False  # посещена ли роботом

    def get_pos(self):
        return [self.x, self.y]


class Graph:
    def __init__(self, ngonArea, step, metric, from_goal, overlap=0.2, start_point=None):
        step *= (1 - overlap)
        self.nodes = []
        self.map = []
        self.node_map = []
        pts = np.array(ngonArea.pts)
        x0 = np.min(pts[:, 0])
        x1 = np.max(pts[:, 0])
        y0 = np.min(pts[:, 1])
        y1 = np.max(pts[:, 1])
        for i, iy in enumerate(np.arange(y0 - 2 * step, y1 + 2 * step, step)):
            map_row = []
            node_map_row = []
            for j, jx in enumerate(np.arange(x0 - 2 * step, x1 + 2 * step, step)):
                tl = [jx - step / 2, iy + step / 2]
                tr = [jx + step / 2, iy + step / 2]
                bl = [jx - step / 2, iy - step / 2]
                br = [jx + step / 2, iy - step / 2]
                if do_contours_intersect([tl, tr, bl, br], pts) or cv2.pointPolygonTest(pts, np.array([jx, iy]),
                                                                                        False) > 0:
                    n = Node(jx, iy)
                    map_row.append(0)
                    node_map_row.append(n)
                    self.nodes.append(n)
                else:
                    map_row.append(1)
                    node_map_row.append(None)
            self.map.append(map_row)
            self.node_map.append(node_map_row)

        # if start_point is not None:
        #     self.nodes.append(Node(*start_point))
        self.map = np.array(self.map)
        self.node_map = np.array(self.node_map)
        self.heuristic = None
        self.visited = None
        self.metric = metric
        self.from_goal = from_goal
        self.max_dist = 0

    def find_nearest_node(self, p):
        def p_dist(n):
            if n is None:
                return np.inf
            return dist(n.get_pos(), p)

        dd = np.vectorize(p_dist)(self.node_map)
        ind = np.unravel_index(dd.argmin(), dd.shape)
        return ind

    def find_farthest_node(self, p):
        def p_dist(n):
            if n is None:
                return -np.inf
            return dist(n.get_pos(), p)

        dd = np.vectorize(p_dist)(self.node_map)
        ind = np.unravel_index(dd.argmax(), dd.shape)
        return ind

    def get_visible_nodes(self, pts):
        # pts = ngon.get_transformed_contour()
        return [n for n in self.nodes if cv2.pointPolygonTest(pts, n.get_pos(), False) > 0]

    def find_route_wavefront(self, pStart):
        for n in self.nodes:
            n.checked = False

        if self.from_goal:
            n = self.find_farthest_node(pStart)
        else:
            n = self.find_nearest_node(pStart)

        if self.heuristic is None:
            self.heuristic = np.full_like(self.node_map, -1, dtype=int)
            self.visited = self.heuristic.copy()
            for i in range(self.heuristic.shape[0]):
                for j in range(self.heuristic.shape[1]):
                    if self.node_map[i, j] is not None:
                        h = self.metric((i, j), n)
                        self.heuristic[i, j] = h
                        self.visited[i, j] = 0
                        self.max_dist = max(h, self.max_dist)

        traj = []
        while True:
            unvisited = self.heuristic[self.visited == 0]
            if unvisited.size == 0:
                break

            curr_dist = np.max(unvisited) if self.from_goal else np.min(unvisited)
            curr_dist_cells = np.argwhere(np.logical_and(self.heuristic == curr_dist, self.visited == 0))

            if traj:
                target = traj[-1]
            else:
                target = pStart
            dd = [dist(target, self.node_map[tuple(c)].get_pos()) for c in curr_dist_cells]
            ind = np.argmin(dd)
            p = curr_dist_cells[ind]
            traj.append(p)
            self.visited[tuple(p)] = 1

            while True:
                neighbours = moore_neighbourhood(traj[-1])
                neighbours = [neigh for neigh in neighbours if (neigh >= [0, 0]).all and
                              (neigh < self.heuristic.shape).all() and
                              self.visited[tuple(neigh)] == 0]
                if not neighbours:
                    break

                if self.from_goal:
                    max_neighs = [neighbours[0]]
                    max_neigh_val = self.heuristic[tuple(neighbours[0])]
                    for neigh in neighbours:
                        if self.heuristic[tuple(neigh)] == max_neigh_val:
                            max_neighs.append(neigh)
                        elif self.heuristic[tuple(neigh)] > max_neigh_val:
                            max_neighs = [neigh]
                            max_neigh_val = self.heuristic[tuple(neigh)]
                    dd = [dist(self.node_map[tuple(traj[-1])].get_pos(), self.node_map[tuple(n)].get_pos()) for n in
                          max_neighs]
                    ind = np.argmin(dd)
                    neigh_to_append = max_neighs[ind]
                else:
                    min_neighs = [neighbours[0]]
                    min_neigh_val = self.heuristic[tuple(neighbours[0])]
                    for neigh in neighbours:
                        if self.heuristic[tuple(neigh)] == min_neigh_val:
                            min_neighs.append(neigh)
                        elif self.heuristic[tuple(neigh)] < min_neigh_val:
                            min_neighs = [neigh]
                            min_neigh_val = self.heuristic[tuple(neigh)]
                    dd = [dist(self.node_map[tuple(traj[-1])].get_pos(), self.node_map[tuple(n)].get_pos()) for n in
                          min_neighs]
                    ind = np.argmin(dd)
                    neigh_to_append = min_neighs[ind]
                traj.append(neigh_to_append)
                self.visited[tuple(neigh_to_append)] = 1

        return [self.node_map[tuple(p)] for p in traj]
