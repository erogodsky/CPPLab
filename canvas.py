import cv2
import numpy as np
from PIL import Image

from helper import Polygon


class Canvas:
    def __init__(self, poly: Polygon, size: int = 600, padding: float = 0.1):
        cv2.namedWindow("canvas", cv2.WINDOW_NORMAL)

        self.min_y = None
        self.min_x = None
        self._poly = poly
        self._padding = int(padding * size)
        self._resize_coef, self.canvas = self._init_canvas(size)

        self.traj = []

        # self._draw_poly(self._poly)

    def _init_canvas(self, size):
        self.min_x, max_x = self._poly.pts[:, 0].min(), self._poly.pts[:, 0].max()
        self.min_y, max_y = self._poly.pts[:, 1].min(), self._poly.pts[:, 1].max()
        len_x = max_x - self.min_x
        len_y = max_y - self.min_y
        resize_coef = size / min(len_x, len_y)
        w, h = int(len_x * resize_coef), int(len_y * resize_coef)
        canvas = np.zeros((h + self._padding * 2, w + self._padding * 2, 3), dtype=np.uint8)
        canvas.fill(255)
        return resize_coef, canvas

    def _natural_coords2image(self, pts):
        pts = np.array(
            [self._resize_coef * (pt - [self.min_x, self.min_y]) + [self._padding, self._padding] for pt in pts],
            dtype=np.int32)
        return pts

    def _image2natural_coords(self, pts):
        pts = np.array(
            [(pt - [self._padding, self._padding]) / self._resize_coef + [self.min_x, self.min_y] for pt in pts],
            dtype=np.float32)
        return pts.reshape(-1, 2)

    def get_image(self):
        self._draw_poly(self._poly)
        self._draw_traj()
        # img = cv2.flip(self.canvas, 1)
        img = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)
        return img

    def show(self):
        img = self.get_image()
        cv2.imshow('canvas', img)
        cv2.waitKey(1)

    def refresh_traj(self, traj):
        self.traj = [(n, True) if n.visited else (n, False) for n in traj]

    def _draw_traj(self):
        for i in range(len(self.traj) - 1):
            p1 = self.traj[i][0].get_pos()
            p2 = self.traj[i + 1][0].get_pos()
            p1, p2 = self._natural_coords2image(np.array([p1, p2]))
            if self.traj[i][1] and self.traj[i+1][1]:
                self.canvas = cv2.line(self.canvas, p1, p2, (0, 255, 0), 2)
            else:
                self.canvas = cv2.line(self.canvas, p1, p2, (0, 0, 255), 2)
            if i == 0:
                cv2.putText(self.canvas, 'Start', p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if i == len(self.traj) - 2:
                cv2.putText(self.canvas, 'End', p2, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def _draw_poly(self, poly: Polygon, color=(255, 0, 255)):
        pts = self._natural_coords2image(poly.pts)
        self.canvas = cv2.polylines(self.canvas, [pts], True, color, 4)
        # self._show()

    def draw_image(self, img, point, orient, real_resolution):
        def rotate_and_paste(image, angle, paste_point, size):
            angle = angle * 180 / np.pi + 180
            paste_point = paste_point.flatten()

            image = Image.fromarray(image)
            image = image.resize((size, size))

            mask = Image.new('L', (size, size), 255)
            image = image.rotate(angle, expand=True)
            mask = mask.rotate(angle, expand=True)
            size = image.size[0]

            canvas = Image.fromarray(self.canvas)
            canvas.paste(image, (paste_point[0] - size // 2, paste_point[1] - size // 2), mask)
            self.canvas = np.array(canvas)

        p1 = point[0] - real_resolution / 2, point[1]
        p2 = point[0] + real_resolution / 2, point[1]
        p1, p2 = self._natural_coords2image(np.array([p1, p2]))
        img_sz = abs(p2[0] - p1[0])
        rotate_and_paste(img, orient, self._natural_coords2image(np.array([point])), img_sz)
        # img = cv2.resize(img, (img_sz, img_sz))
        # rot_mat = cv2.getRotationMatrix2D(point, orient, img.shape[0] / img_sz)
        # img = cv2.warpAffine(img, rot_mat, (img_sz, img_sz))
        # self._show()

    def draw_grid(self, graph):
        for n in graph.nodes:
            color = (200, 100, 50) if n.observed else (0, 0, 0)
            r = 8 if n.visited else 5
            coords = self._natural_coords2image(np.array([[n.x, n.y]])).flatten()
            self.canvas = cv2.circle(self.canvas, coords, r, color, -1)
        self.show()

    def draw_text(self, text, coord):
        self.canvas = cv2.putText(self.canvas, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
