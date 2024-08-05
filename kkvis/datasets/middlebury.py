import os

import cv2
import numpy as np


def parse_matlab_array_string(string):
    string = string.replace(" ", ",")
    string = string.replace(";", "")
    string = string.replace("[", "")
    string = string.replace("]", "")

    string = string.split(",")

    return np.asarray(string, dtype=np.float32)


class MiddleburyDataset:
    def __init__(self, data_path):

        # Load calibration file
        fp = os.path.join(data_path, "calib.txt")
        file = open(fp, mode="r")
        lines = file.read()
        lines = lines.splitlines()
        file.close()
        data_dict = {}

        for idx, line in enumerate(lines):
            line = line.split("=")
            if idx < 2:
                # Change to size (3,3)
                value = parse_matlab_array_string(line[1]).reshape(3, 3)
            else:
                value = np.float32(line[1])
            data_dict[line[0]] = value

        self.calib = data_dict
        self.K0 = self.calib["cam0"]
        self.K1 = self.calib["cam1"]

        # Load images
        self.im0 = cv2.imread(os.path.join(data_path, "im0.png"), cv2.IMREAD_COLOR)
        self.im1 = cv2.imread(os.path.join(data_path, "im1.png"), cv2.IMREAD_COLOR)


if __name__ == "__main__":
    motorbike = MiddleburyDataset(
        "/Users/kenton/projects/kklo-ml/data/middlebury/motorcycle"
    )
