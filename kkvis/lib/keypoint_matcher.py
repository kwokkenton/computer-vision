import cv2
import numpy as np


class KeypointMatcher:
    """This class detect and compute keypoints and descriptors
    from images using the class orb object
    """

    def __init__(self):
        self.kp_extractor = cv2.ORB_create(3000)

        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(
            indexParams=index_params, searchParams=search_params
        )
        return

    def get_matches(self, im0, im1, show=True):
        """

        Note keypoint matches are in (y,x) convention.

        Args:
            im1 (np.ndarray[np.uint8]): Source Image, A 3D array of with shape (H,W,3)
            im2 (np.ndarray[np.uint8]): Destination Image, A 3D array of with shape
            (H,W,3)

        Returns:
            q1 (np.ndarray[np.float32]): The N good keypoints matches position in im0
                shape (N,2)
            q2 (np.ndarray[np.float32]): The N good keypoints matches position in im1
                shape (N,2)
        """
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.kp_extractor.detectAndCompute(im0, None)
        kp2, des2 = self.kp_extractor.detectAndCompute(im1, None)

        # Find matches
        matches = self.matcher.knnMatch(des1, des2, k=2)

        # Find the matches that do not have a too high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        if show:
            draw_params = dict(
                matchColor=-1,  # draw matches in green color
                singlePointColor=None,
                matchesMask=None,  # draw only inliers
                flags=2,
            )

            img3 = cv2.drawMatches(im0, kp1, im1, kp2, good, None, **draw_params)
            cv2.imshow("image", img3)
            cv2.waitKey(0)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2
