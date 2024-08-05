import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d

from kkvis.lib.vision import form_transf

WIDTH = 1080
HEIGHT = 720


class CVSimulation:
    def __init__(self, data, cameras=None):
        self.data = data
        self.cameras = cameras
        self.viz = None

    def visualise(self):
        self.viz = open3d.visualization.Visualizer()
        self.viz.create_window(width=WIDTH, height=HEIGHT)
        pcd = open3d.geometry.PointCloud()
        pcd.points.extend(self.data.points)
        self.viz.add_geometry(pcd)

        # The x, y, z axis will be rendered as red, green, and blue arrows respectively.
        axis = open3d.geometry.TriangleMesh().create_coordinate_frame()
        self.viz.add_geometry(axis)

        for cam in self.cameras:
            # Add estimated camera
            cameraLines = open3d.geometry.LineSet.create_camera_visualization(
                view_width_px=cam.width,
                view_height_px=cam.height,
                intrinsic=cam.K,
                extrinsic=cam.pose,
            )
            camera_axis = open3d.geometry.TriangleMesh().create_coordinate_frame()
            # Camera transformation is inverse of coordinate transformation
            camera_axis.transform(np.linalg.inv(cam.pose))
            self.viz.add_geometry(cameraLines)
            self.viz.add_geometry(camera_axis)

        self.viz.run()
        return


class Camera:
    def __init__(self, K, R, t, width, height):
        # t: from camera to Origin
        self.K = K
        self.R = R
        self.t = t
        self.pose = form_transf(R, t)
        self.width = width
        self.height = height
        self.projected_points = None

    def project_points(self, points, show=True):
        # this computes the image coordinates of the points as they are
        # projected onto the image plane of the camera.
        projected_points, _ = cv2.projectPoints(points, self.R, self.t, self.K, None)

        if show:
            plt.title("Projected points for camera (image coordinates)")
            plt.scatter(projected_points.T[0], projected_points.T[1])
            plt.xlim(0, self.width)
            plt.ylim(self.height, 0)
            plt.xlabel("x")
            plt.ylabel("y")

            plt.show()

        self.projected_points = np.squeeze(projected_points)
        return self.projected_points


class Simulated3DDataset:
    def __init__(self, N_points):
        np.random.seed(0)
        self.points = np.random.rand(N_points, 3)
