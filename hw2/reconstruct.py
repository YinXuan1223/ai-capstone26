import os
import re
import glob
import numpy as np
import open3d as o3d
import argparse
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import time
import json

# ---------- Camera Intrinsics (Resolution 512x512, FOV 90) ----------
# These parameters are derived from the Habitat pinhole camera model [cite: 26-27].
IMG_W, IMG_H = 512, 512
FOV = np.deg2rad(90.0)
FX = (IMG_W / 2.0) / np.tan(FOV / 2.0)
FY = (IMG_H / 2.0) / np.tan(FOV / 2.0)
CX, CY = IMG_W / 2.0, IMG_H / 2.0
DEPTH_SCALE = 1000.0 #

def depth_image_to_point_cloud(rgb_image, depth_image):

    """
    TASK 1: Geometric Unprojection [cite: 12, 25-27]
    Convert depth pixels (u, v, d) into 3D world points (x, y, z).
    """
    # 1. Convert inputs to numpy arrays
    # 2. Convert depth to meters (Habitat depth is often scaled or normalized)
    rgb = np.asarray(rgb_image).astype(np.float32) / 255.0
    depth = np.asarray(depth_image).astype(np.float32)
    depth_m = depth / 255.0 * 10.0

    # 3. Create a coordinate grid for (u, v) pixels
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Implement unprojection logic here
    z = depth_m
    x = (u - CX) * z / FX
    y = -(v - CY) * z / FY
    points = np.stack((x, y, -z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)

    valid = z.flatten() > 0
    points = points[valid]
    colors = colors[valid]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd # PointCloud with 262144 points.

def preprocess_point_cloud(pcd, voxel_size):

    """
    Pre-processing: Voxelization and Normal Estimation [cite: 17, 29]
    """

    pcd_down = pcd.voxel_down_sample(voxel_size) # built-in function
    
    # Estimate normals for pcd_down (required for Point-to-Plane ICP)
    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30) # k-dimensional tree
    )
    
    # Compute FPFH features for Global Registration [cite: 30] Fast Point Feature Histograms
    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature( # Feature class with dimension = 33
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000, 0.999)
    )
    return result.transformation

def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """
    TASK 2: Open3D ICP Implementation (REQUIRED) [cite: 32]
    """
    # Use o3d.pipelines.registration.registration_icp
    # Estimation method should be TransformationEstimationPointToPlane()

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    return result.transformation

def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, gt_poses, args):
    """
    TASK 3: Evaluation & Visualization [cite: 19, 35-38]
    """
    # 1. Create LineSet for estimated trajectory (Red)
    # 2. Create LineSet for ground truth trajectory (Black)

    pred_xyz = np.array([p[:3, 3] for p in predicted_cam_poses])
    gt_xyz = np.array([p[:3, 3] for p in gt_poses])

    # 建立預測軌跡線段 (紅色)
    points = pred_xyz
    lines = [[i, i+1] for i in range(len(points)-1)]
    line_set_pred = o3d.geometry.LineSet()
    line_set_pred.points = o3d.utility.Vector3dVector(points)
    line_set_pred.lines = o3d.utility.Vector2iVector(lines)
    line_set_pred.paint_uniform_color([1, 0, 0]) # Red

    # 建立 GT 軌跡線段 (黑色)
    points_gt = gt_xyz
    line_set_gt = o3d.geometry.LineSet()
    line_set_gt.points = o3d.utility.Vector3dVector(points_gt)
    line_set_gt.lines = o3d.utility.Vector2iVector(lines)
    line_set_gt.paint_uniform_color([0, 0, 0]) # Black

    


    # Calculate Mean L2 Distance between predicted_cam_poses and gt_poses [cite: 38]
    # L2 = sqrt(dx^2 + dy^2 + dz^2)
    mean_l2_error = 0.0 
    errors = []
    for pred, gt in zip(predicted_cam_poses, gt_poses):
        pred_t = pred[:3, 3]
        gt_t = gt[:3, 3]
        errors.append(np.linalg.norm(pred_t - gt_t))

    mean_l2_error = np.mean(errors)
    print(f"Mean L2 distance: {mean_l2_error:.6f} meters")
    
    o3d.visualization.draw_geometries(
        [reconstructed_pcd, line_set_pred, line_set_gt],window_name=f"Floor {args.floor} Reconstruction"
        # [reconstructed_pcd],window_name=f"Floor {args.floor} Reconstruction"
    )


def reconstruct(args):

    voxel_size = 0.08

    rgb_dir = os.path.join(args.data_root, "rgb")
    depth_dir = os.path.join(args.data_root, "depth")
    rgb_files   = sorted(glob.glob(os.path.join(rgb_dir,   "*.png")), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    
    # Load Ground Truth Poses [cite: 24, 54]
    gt_pose_path = os.path.join(args.data_root, "GT_pose.npy")
    gt_poses = []
    if os.path.exists(gt_pose_path):
        gt_data = np.load(gt_pose_path)
        for p in gt_data:
            mat = np.eye(4)
            mat[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
            mat[:3, 3] = [p[0], p[1], p[2]]
            gt_poses.append(mat)
        gt_poses = np.stack(gt_poses)
    
    cam0_rgb = o3d.io.read_image(rgb_files[0])
    cam0_depth = o3d.io.read_image(depth_files[0])
    cam0_pcd = depth_image_to_point_cloud(cam0_rgb, cam0_depth)

    cam0_to_world = gt_poses[0].copy()
    pcd0 = deepcopy(cam0_pcd)
    pcd0.transform(cam0_to_world)

    accumulated_pcd = pcd0
    camera_poses = [cam0_to_world] # camera_poses[-1] is the transformation from cam_target to world
    

    # Reconstruction Loop [cite: 29-30]
    for i in range(1, len(rgb_files)):

        print(f"Processing Frame {i}...")

        # Implement the full pipeline:
        source_rgb = o3d.io.read_image(rgb_files[i])
        source_depth = o3d.io.read_image(depth_files[i])
        target_rgb = o3d.io.read_image(rgb_files[i-1])
        target_depth = o3d.io.read_image(depth_files[i-1])
        
        # 1. Convert RGB-D to PointCloud (Task 1)
        source_pcd = depth_image_to_point_cloud(source_rgb, source_depth) # (262144, 3)
        target_pcd = depth_image_to_point_cloud(target_rgb, target_depth)
        
        # 2. Preprocess (Voxel/FPFH/Normals)
        source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
        
        # 3. Execute Global Registration (RANSAC)
        trans_init = execute_global_registration(
            source_down, target_down,
            source_fpfh, target_fpfh,
            voxel_size
        )

        # 4. Execute Local Registration (ICP - Task 2)
        trans_icp = local_icp_algorithm(source_down, target_down, trans_init, voxel_size * 1.5)
    
        # 5. Update camera_poses and accumulate points
        cam_source_to_world = camera_poses[-1] @ trans_icp
        camera_poses.append(cam_source_to_world)

        source_pcd_world = deepcopy(source_pcd)
        source_pcd_world.transform(cam_source_to_world)
        accumulated_pcd += source_pcd_world


    
    # Post-processing: remove the ceiling [cite: 37]
    points = np.asarray(accumulated_pcd.points)
    y = points[:, 1]
    print(f"y min: {y.min():.4f}, y max: {y.max():.4f}, range: {y.max() - y.min():.4f}")

    mask = points[:, 1] < 0.5 # 假設 0.5 公尺以上是天花板
    accumulated_pcd = accumulated_pcd.select_by_index(np.where(mask)[0])

    return accumulated_pcd, camera_poses, gt_poses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    args = parser.parse_args()

    # Set data root based on floor
    args.data_root = f"data_collection/first_floor/" if args.floor == 1 else f"data_collection/second_floor/"

    start_time = time.time()
    result_pcd, pred_poses, gt_poses = reconstruct(args)
    
    print(f"new Total execution time: {time.time() - start_time:.2f}s") # 
    visualize_and_evaluate(result_pcd, pred_poses, gt_poses, args)