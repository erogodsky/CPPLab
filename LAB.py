import cv2
from pathlib import Path

from drone import experiment


def chebyshev():
    def metric(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        h = max(abs(x1 - x2), abs(y1 - y2))
        return h

    def neighbours(p):
        return [
            (p[0] - 1, p[1]),
            (p[0] + 1, p[1]),
            (p[0], p[1] - 1),
            (p[0], p[1] + 1)
        ]

    return metric, neighbours


def manhatten():
    def metric(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        h = abs(x1 - x2) + abs(y1 - y2)
        return h

    def neighbours(p):
        return [
            (p[0] - 1, p[1] - 1),
            (p[0] + 1, p[1] + 1),
            (p[0] + 1, p[1] - 1),
            (p[0] - 1, p[1] + 1)
        ]

    return metric, neighbours


def vertical():
    def metric(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        h = abs(y1 - y2)
        return h

    def neighbours(p):
        return [
            (p[0], p[1] - 1),
            (p[0], p[1] + 1)
        ]

    return metric, neighbours


def horizontal():
    def metric(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        h = abs(x1 - x2)
        return h

    def neighbours(p):
        return [
            (p[0] - 1, p[1]),
            (p[0] + 1, p[1])
        ]

    return metric, neighbours


metrics = {'chebyshev': chebyshev, 'manhatten': manhatten, 'vertical': vertical, 'horizontal': horizontal}

path = Path("exp10")
path.mkdir(exist_ok=True)

for metric_name in metrics:
    for from_goal in (True, False):
        speed, img = experiment(chebyshev, True)
        with open(str(path / 'results.txt'), 'a+') as f:
            f.write(f"{metric_name}, {'from goal' if from_goal else 'from UAV'}: {str(speed)}\n")
        cv2.imwrite(str(path / f"{metric_name}_{'from goal' if from_goal else 'from UAV'}.png"), img)
