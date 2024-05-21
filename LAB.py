import cv2
from pathlib import Path

from drone import experiment


def chebyshev(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    h = max(abs(x1 - x2), abs(y1 - y2))
    return h


metrics = {
    'chebyshev': chebyshev,
}

path = Path("exp10")
path.mkdir(exist_ok=True)

for metric_name in metrics:
    for from_goal in (True, False):
        speed, img = experiment(metric_name, from_goal)
        with open(str(path / 'results.txt'), 'a+') as f:
            f.write(f"{metric_name}, {'from goal' if from_goal else 'from UAV'}: {str(speed)}\n")
        cv2.imwrite(str(path / f"{metric_name}_{'from goal' if from_goal else 'from UAV'}.png"), img)
