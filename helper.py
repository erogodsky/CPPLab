import time
import numpy as np


class Polygon:
    def __init__(self, pts):
        self.pts = np.array(pts, dtype=np.float32)
        self.ang = 0
        self.x = 0
        self.y = 0


def dist(n1, n2):
    return np.linalg.norm((n1[0] - n2[0], n1[1] - n2[1]))


def do_segments_intersect(s0, s1):
    w0 = s0[1][0] - s0[0][0]
    w1 = s1[1][0] - s1[0][0]
    h0 = s0[1][1] - s0[0][1]
    h1 = s1[1][1] - s1[0][1]
    # векторные произведения (площади треугольников, образованных из сочетаний точек отрезков)
    p0 = h1 * (s1[1][0] - s0[0][0]) - w1 * (s1[1][1] - s0[0][1])
    p1 = h1 * (s1[1][0] - s0[1][0]) - w1 * (s1[1][1] - s0[1][1])
    p2 = h0 * (s0[1][0] - s1[0][0]) - w0 * (s0[1][1] - s1[0][1])
    p3 = h0 * (s0[1][0] - s1[1][0]) - w0 * (s0[1][1] - s1[1][1])
    return p0 * p1 <= 0 and p2 * p3 <= 0


def do_contours_intersect(contour1, contour2):
    for i in range(len(contour1)):
        p1 = contour1[i]
        p2 = contour1[(i + 1) % len(contour1)]

        for j in range(len(contour2)):
            q1 = contour2[j]
            q2 = contour2[(j + 1) % len(contour2)]

            if do_segments_intersect([p1, p2], [q1, q2]):
                return True

    return False


def moore_neighbourhood(p):
    return np.array([
        [p[0] - 1, p[1] - 1],
        [p[0] - 1, p[1]],
        [p[0] - 1, p[1] + 1],
        [p[0], p[1] - 1],
        [p[0], p[1] + 1],
        [p[0] + 1, p[1] - 1],
        [p[0] + 1, p[1]],
        [p[0] + 1, p[1] + 1],
    ])


def interpolate_pos(traj, L, target, drone):
    LSum = 0
    for i in range(1, len(traj)):
        p1, p2 = traj[i - 1], traj[i]
        step = dist(p1, p2)
        LSum += step
        if LSum > L:
            if target is None:
                target = p2
            else:
                if p2 != target:
                    time.sleep(4)
                    drone.shoot()
                    target = p2
            dL = LSum - L
            k1 = dL / step
            k2 = 1 - k1
            return np.array(p1) * k1 + np.array(p2) * k2, target
    if L < 0:
        return traj[0], target
    time.sleep(4)
    drone.shoot()
    return traj[-1], target
