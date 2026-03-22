import cv2
import numpy as np

points = []

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        # camera intrinsic parameters
        w = 512
        h = 512
        u0 = w / 2.0
        v0 = h / 2.0
        fx = u0 / np.tan(np.deg2rad(fov / 2.0))
        fy = v0 / np.tan(np.deg2rad(fov / 2.0))

        # camera extrinsic parameters
        C_front = np.array([0.0, 1.0, 0.0])
        C_bev = np.array([0.0, 2.5, 0.0])

        # camera coordinate is obtained from rotating world coordinate by th_BEV around x-axis (in chinese: how world coordinate do to be camera coordinate)
        th_BEV = np.deg2rad(90)   
        Rx_BEV = np.array([
            [1, 0, 0],
            [0, np.cos(th_BEV), -np.sin(th_BEV)],
            [0, np.sin(th_BEV),  np.cos(th_BEV)]
        ], dtype=np.float64)
        # front camera coordinate is obtained from rotating world coordinate by th_front around x-axis
        th_front = np.deg2rad(180)
        Rx_front = np.array([
            [1, 0, 0],
            [0, np.cos(th_front), -np.sin(th_front)],
            [0, np.sin(th_front),  np.cos(th_front)]
        ], dtype=np.float64)

        new_pixels = []

        for p in points:

            u, v = p[0], p[1]
            x_cam = (u - u0) / fx
            y_cam = (v - v0) / fy

            ray_cam = np.array([x_cam, y_cam, 1.0], dtype=np.float64)
            ray_world = Rx_BEV @ ray_cam

            lam = -C_bev[1] / ray_world[1]
            P_world = C_bev + lam * ray_world

            P_cam = np.linalg.inv(Rx_front) @ (P_world - C_front)
            X, Y, Z = P_cam
            u_front = fx * (X / Z) + u0
            v_front = fy * (Y / Z) + v0

            new_pixels.append([int(u_front), int(v_front)])

        return new_pixels


    def top_to_front_all_matrices(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """     
        # camera intrinsic parameters
        w = 512
        h = 512
        u0 = w / 2.0
        v0 = h / 2.0
        fx = u0 / np.tan(np.deg2rad(fov / 2.0))
        fy = v0 / np.tan(np.deg2rad(fov / 2.0))

        # camera extrinsic parameters
        C_front = np.array([0.0, 1.0, 0.0])
        C_bev = np.array([0.0, 2.5, 0.0])

        # camera intrinsic matrix
        InMat = np.array([
            [fx, 0, u0],
            [0, fy, v0],
            [0, 0,   1]
        ], dtype=np.float64)

        # camera extrinsic matrices
        ExtMat_BEV = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 2.5],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        ExtMat_front = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 1],
            [0, 0, -1, 0]
        ], dtype=np.float64)

        new_pixels = []

        for p in points:

            u, v = p[0], p[1]

            Z_BEVcam = 2.5
            X_BEVcam = (u-u0)*2.5/fx
            Y_BEVcam = (v-v0)*2.5/fy
            P_BEVcam = np.array([X_BEVcam, Y_BEVcam, Z_BEVcam, 1.0])

            P_world = np.linalg.inv(ExtMat_BEV) @ P_BEVcam
            front_img_homo = InMat @ ExtMat_front @ P_world

            u_front = front_img_homo[0]/front_img_homo[2]
            v_front = front_img_homo[1]/front_img_homo[2]

            new_pixels.append([int(u_front), int(v_front)])

        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


if __name__ == "__main__":

    pitch_ang = -90

    front_rgb = "bev_data/front1.png"
    top_rgb = "bev_data/bev1.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels)