from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import cv2
import numpy as np
import time

from canvas import Canvas
from graph import Graph
from helper import interpolate_pos, Polygon


class Drone:
    def __init__(self, sim, poly: Polygon, height: float):
        self.field_size = 2 * height * np.tan(np.pi / 6 / 2)

        self.sim = sim
        self.handle = sim.getObject("./Quadcopter")
        self.vision_sensor = sim.getObject("./Quadcopter/Vision_sensor")

        self.target = sim.getObject("./target")

        self.height = height
        self.get_high(self.height)

        self.canvas = Canvas(poly, padding=0.3)
        self.segm_canvas = Canvas(poly)
        self.images = []

    def get_field(self):
        def rotate_point(point, center, angle):
            # Вычисляем синус и косинус угла поворота
            sin_a = np.sin(angle)
            cos_a = np.cos(angle)

            # Получаем координаты точки и центра
            x, y = point
            cx, cy = center

            # Поворачиваем точку относительно центра
            new_x = cos_a * (x - cx) - sin_a * (y - cy) + cx
            new_y = sin_a * (x - cx) + cos_a * (y - cy) + cy

            return np.array([new_x, new_y], dtype=np.float32)

        center = self.get_position()[:2]
        pts = np.array([[center[0] - self.field_size / 2, center[1] + self.field_size / 2],
                        [center[0] + self.field_size / 2, center[1] + self.field_size / 2],
                        [center[0] + self.field_size / 2, center[1] - self.field_size / 2],
                        [center[0] - self.field_size / 2, center[1] - self.field_size / 2]])
        angle = self.get_sensor_orientation()
        pts = np.array([rotate_point(p, center, -angle) for p in pts])

        return pts

    def get_high(self, h: float):
        act_h = self.sim.getObjectPosition(self.target, self.sim.handle_world)[2]
        while act_h < h:
            act_h += 0.5
            time.sleep(0.1)
            self.sim.setObjectPosition(self.target, self.sim.handle_world, [*self.get_position()[:2], act_h])
        time.sleep(1)

    def set_target_pos(self, coord):
        self.sim.setObjectPosition(self.target, self.sim.handle_world, [*coord, self.height])

    def shoot(self):
        image, resolution = self.sim.getVisionSensorImg(self.vision_sensor)
        image = np.frombuffer(image, dtype=np.uint8)
        image = np.array(image, dtype=np.uint8).reshape((*resolution, 3))

        x, y, _ = self.get_position()
        orient = self.get_sensor_orientation()
        self.canvas.draw_image(image, (x, y), orient, self.field_size)
        self.images.append(image)

    def get_sensor_orientation(self):
        return self.sim.getObjectOrientation(self.vision_sensor, self.sim.handle_world)[2]

    def get_position(self):
        self.x, self.y, self.z = self.sim.getObjectPosition(self.handle, self.sim.handle_world)
        return np.array([self.x, self.y, self.z])


def experiment(metric, from_goal):
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    sim.setObjectPosition(sim.getObject("./target"), sim.getObject("./Quadcopter"), [0, 0, 0])
    sim.startSimulation()

    markers = sim.getObject("./markers")
    poly = []
    for m in sim.getObjectsInTree(markers, sim.handle_all, 1):
        poly.append(sim.getObjectPosition(m, -1)[:2])
    poly = Polygon(poly)

    drone = Drone(sim, poly, 30)
    start_pos = drone.get_position()

    overlap = 0.3
    graph = Graph(poly, drone.field_size, metric, from_goal, overlap=overlap)
    drone.canvas.draw_grid(graph)

    s = time.time()
    route = graph.find_route_wavefront(drone.get_position()[:2])
    print(time.time() - s)
    traj = [n.get_pos() for n in route]

    area = (drone.field_size * (1 - overlap)) ** 2 * len(traj)

    traj.insert(0, drone.get_position()[:2])

    t_sim = 0
    v_drone = 5

    target = None
    p = [0, 0]

    saved = None

    while True:
        for n in graph.nodes:
            n.visited = n.observed = False

        p_new, target = interpolate_pos(traj, v_drone * t_sim, target, drone)
        if target == traj[2]:
            t_start = time.time()
        if abs(p_new[0] - p[0]) < 0.001 and abs(p_new[1] - p[1]) < 0.001:
            break
        else:
            p = p_new

        drone.canvas.show()

        drone.set_target_pos(p)
        nearest_node = graph.node_map[graph.find_nearest_node(p)]
        nearest_node.visited = True
        drone.canvas.refresh_traj(route)

        if saved is None:
            saved = drone.canvas.get_image()

        vis_nodes = graph.get_visible_nodes(drone.get_field())

        for n in vis_nodes:
            n.observed = True

        drone.canvas.show()

        t_sim += 0.1

    t_end = time.time()
    v = area / (t_end - t_start)

    sim.setObjectPosition(drone.handle, sim.handle_world, [*start_pos, drone.height])
    sim.stopSimulation()

    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, result = stitcher.stitch([img[0] for img in drone.images])
    cv2.namedWindow("stitch", cv2.WINDOW_NORMAL)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = cv2.flip(result, 0)
    cv2.imshow('stitch', result)
    drone.segm_canvas.show()
    cv2.waitKey(0)

    return v, saved
