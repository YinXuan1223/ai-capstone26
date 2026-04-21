import os
import glob
import numpy as np
import open3d as o3d
import argparse
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import time

# ---------- Camera Intrinsics (Resolution 512x512, FOV 90) ----------
IMG_W, IMG_H = 512, 512
FOV = np.deg2rad(90.0)
FX = (IMG_W / 2.0) / np.tan(FOV / 2.0)
FY = (IMG_H / 2.0) / np.tan(FOV / 2.0)
CX, CY = IMG_W / 2.0, IMG_H / 2.0
DEPTH_SCALE = 1000.0

start = time.time()

# ─────────────────────────────────────────────
# TASK 1: Depth Image → Point Cloud
# ─────────────────────────────────────────────
def depth_image_to_point_cloud(rgb_image, depth_image):
    """
    Convert an RGB-D frame into an Open3D PointCloud.

    Unprojection formula (pinhole model, camera looks toward -Z):
        x =  (u - CX) * z / FX
        y = -(v - CY) * z / FY   ← flip Y so Y-up in camera space
        z = -depth                ← camera looks toward -Z
    """
    rgb = np.asarray(rgb_image, dtype=np.float32)
    depth = np.asarray(depth_image, dtype=np.float32)

    # --- depth scaling -------------------------------------------------
    # uint16 or large values → millimetres stored as integers
    if depth.dtype == np.uint16 or np.max(depth) > 255:
        depth_m = depth / DEPTH_SCALE
    else:
        # 8-bit fallback: 0-255 mapped to 0-10 m
        depth_m = depth / 255.0 * 10.0

    # --- pixel grid ----------------------------------------------------
    u, v = np.meshgrid(
        np.arange(IMG_W, dtype=np.float32),
        np.arange(IMG_H, dtype=np.float32),
    )

    z = depth_m
    valid = np.isfinite(z) & (z > 0.05)          # discard invalid / too-near

    x =  (u - CX) * z / FX
    y = -(v - CY) * z / FY                        # Y-up
    points = np.stack([x, y, -z], axis=-1)[valid] # Z-forward → negate

    colors = rgb[valid] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


# ─────────────────────────────────────────────
# Pre-processing: Voxelisation + Normals + FPFH
# ─────────────────────────────────────────────
def preprocess_point_cloud(pcd, voxel_size):
    """Downsample, estimate normals, compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)

    if len(pcd_down.points) == 0:
        return pcd_down, o3d.pipelines.registration.Feature()

    # Normals – required for Point-to-Plane ICP
    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # FPFH features for global registration
    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


# ─────────────────────────────────────────────
# Global Registration (RANSAC)
# ─────────────────────────────────────────────
def execute_global_registration(source_down, target_down,
                                source_fpfh, target_fpfh, voxel_size):
    """RANSAC-based feature matching for an initial coarse alignment."""
    if len(source_down.points) < 4 or len(target_down.points) < 4:
        return None

    dist_thresh = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000, 0.999),
    )
    return result


# ─────────────────────────────────────────────
# TASK 2 (Required): Open3D ICP
# ─────────────────────────────────────────────
def local_icp_algorithm(source_down, target_down, trans_init, threshold):

    """
    Point-to-Plane ICP via Open3D.
    Falls back to Point-to-Point if normals are missing or the call raises.
    """
    # Guarantee normals exist on both clouds
    for cloud in (source_down, target_down):
        if not cloud.has_normals():
            cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2.0, max_nn=30)
            )

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)

    try:
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down,
            threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria,
        )
    except RuntimeError:
        # Degenerate geometry: fall back to Point-to-Point
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down,
            threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria,
        )
    return result


# ─────────────────────────────────────────────
# TASK 2 (Bonus): Custom ICP
# ─────────────────────────────────────────────

def _nearest_neighbor(source_pts, target_pts, chunk=1024):
    """Brute-force nearest-neighbour search (chunked to limit peak RAM)."""
    target_tree = o3d.geometry.KDTreeFlann(target_pts)
    
    indices = []
    distances = []
    for idx, pt in enumerate(source_pts):
        [k, ind, dis] = target_tree.search_knn_vector_3d(pt, 1)
        indices.append(ind)
        distances.append(np.sqrt(dis))

    return np.array(indices).flatten(), np.array(distances).flatten()


def _estimate_rigid_transform(src, dst):
    """SVD-based rigid transform: find R, t such that dst ≈ R @ src + t."""
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    rot = Vt.T @ U.T
    if np.linalg.det(rot) < 0:          # reflection check
        Vt[-1] *= -1
        rot = Vt.T @ U.T
    t = dst_c - rot @ src_c
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T


def my_local_icp_algorithm(source_pcd, target_pcd, initial_transform,
                            voxel_size=0.25, max_iter=50):
    """
    Custom Point-to-Point ICP with trimmed correspondences.

    Steps per iteration:
      1. Apply current transform to source points.
      2. Find nearest neighbours in target.
      3. Filter by distance threshold; trim top-10% outliers.
      4. Estimate rigid transform with SVD and accumulate.
      5. Stop when mean error change < 1e-5.
    """
    src_pts = np.asarray(source_pcd.points, dtype=np.float64)
    tgt_pts = np.asarray(target_pcd.points, dtype=np.float64)

    if src_pts.shape[0] == 0 or tgt_pts.shape[0] == 0:
        res = o3d.pipelines.registration.RegistrationResult()
        res.transformation = initial_transform.copy()
        return res

    # Sub-sample source if too large (speed)
    if src_pts.shape[0] > 2000:
        idx = np.linspace(0, src_pts.shape[0] - 1, 2000, dtype=np.int32)
        src_pts = src_pts[idx]

    T = initial_transform.copy().astype(np.float64)
    corr_thresh = max(voxel_size * 2.0, 1e-3)
    prev_err = np.inf
    best_fitness, best_rmse = 0.0, np.inf

    for _ in range(max_iter):
        # 1. Transform source
        rot, trans = T[:3, :3], T[:3, 3]
        transformed = src_pts @ rot.T + trans

        # 2. Nearest neighbours
        nn_idx, nn_dist = _nearest_neighbor(transformed, target_pcd)

        # 3. Distance filter
        valid = nn_dist < corr_thresh
        if valid.sum() < 6:
            corr_thresh *= 1.25
            if valid.sum() < 3:
                continue
        
        valid_dist = nn_dist[valid]
        trim = np.percentile(valid_dist, 90)
        inlier = valid.copy()
        inlier[valid] = valid_dist <= trim
        if inlier.sum() < 3:
            inlier = valid
        if inlier.sum() < 3:
            break

        # 4. SVD rigid transform
        matched_src = transformed[inlier]
        matched_tgt = tgt_pts[nn_idx[inlier]]
        delta = _estimate_rigid_transform(matched_src, matched_tgt)
        T = delta @ T

        # 5. Convergence check
        mean_err = float(np.mean(nn_dist[inlier]))
        best_fitness = max(best_fitness, inlier.sum() / src_pts.shape[0])
        best_rmse = min(best_rmse, float(np.sqrt(np.mean(nn_dist[inlier] ** 2))))
        if abs(prev_err - mean_err) < 1e-5:
            break
        prev_err = mean_err

    res = o3d.pipelines.registration.RegistrationResult()
    res.transformation = T
    # fitness / inlier_rmse are read-only in some o3d versions, so store them separately
    res.fitness = best_fitness
    res.inlier_rmse = best_rmse
    return res


# ─────────────────────────────────────────────
# Ceiling Removal
# ─────────────────────────────────────────────
def remove_ceiling(pcd, percentile=98.0, margin=0.05):
    """Remove the top 'percentile' Y-values (ceiling points)."""
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return pcd
    cutoff = np.percentile(pts[:, 1], percentile) - margin
    keep = pts[:, 1] < cutoff
    filtered = o3d.geometry.PointCloud()
    filtered.points = o3d.utility.Vector3dVector(pts[keep])
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
        filtered.colors = o3d.utility.Vector3dVector(cols[keep])
    return filtered


# ─────────────────────────────────────────────
# TASK 3: Evaluation & Visualisation
# ─────────────────────────────────────────────
def _make_lineset(positions, color):
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[0] < 2:
        return None
    lines = [[i, i + 1] for i in range(positions.shape[0] - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(positions)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color, dtype=np.float64), (len(lines), 1))
    )
    return ls


def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, gt_poses, args):
    """
    Compute Mean L2 distance between predicted and GT camera positions,
    then launch an Open3D viewer with both trajectories overlaid.
    """
    pred_pos = np.asarray([T[:3, 3] for T in predicted_cam_poses], dtype=np.float64)
    gt_pos   = np.asarray([T[:3, 3] for T in gt_poses],            dtype=np.float64)

    n = min(len(pred_pos), len(gt_pos))

    # Mean L2 distance
    l2_per_frame = np.linalg.norm(pred_pos[:n] - gt_pos[:n], axis=1)
    mean_l2_error = float(l2_per_frame.mean())
    print(f"Mean L2 distance: {mean_l2_error:.6f} meters")

    # Build line sets
    pred_ls = _make_lineset(pred_pos[:n], [1.0, 0.0, 0.0])   # red
    gt_ls   = _make_lineset(gt_pos[:n],   [0.0, 0.0, 0.0])   # black

    geometries = [reconstructed_pcd]
    if pred_ls is not None:
        geometries.append(pred_ls)
    if gt_ls is not None:
        geometries.append(gt_ls)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Floor {args.floor} Reconstruction  |  L2={mean_l2_error:.4f} m",
    )
    return mean_l2_error


# ─────────────────────────────────────────────
# Main Reconstruction Loop
# ─────────────────────────────────────────────
def _load_gt_poses(gt_pose_path):
    """Load GT_pose.npy → list of 4×4 homogeneous matrices."""
    gt_data = np.load(gt_pose_path)            # (N, 7)  [tx ty tz qw qx qy qz]
    mats = []
    for p in gt_data:
        mat = np.eye(4, dtype=np.float64)
        # scipy expects [qx, qy, qz, qw]; GT stores [qw, qx, qy, qz]
        mat[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
        mat[:3, 3]  = p[:3]
        mats.append(mat)
    return mats


def _register_pair(src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size, version):
    """Global (RANSAC) → Local (ICP) registration for one frame pair."""
    # --- global init ---
    global_result = execute_global_registration(
        src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size
    )
    trans_init = np.eye(4, dtype=np.float64)
    if global_result is not None and getattr(global_result, "fitness", 0.0) > 1e-3:
        trans_init = global_result.transformation

    threshold = voxel_size * 1.5

    # --- local refinement ---
    if version == "my_icp":
        local_result = my_local_icp_algorithm(
            src_down, tgt_down, trans_init, voxel_size
        )
        # fallback if my_icp fails
        if getattr(local_result, "_fitness", getattr(local_result, "fitness", 0.0)) < 1e-3:
            print("my_icp_fails!")
            local_result = my_local_icp_algorithm(
                src_down, tgt_down, np.eye(4, dtype=np.float64), voxel_size
            )
    else:  # "open3d"
        local_result = local_icp_algorithm(src_down, tgt_down, trans_init, threshold)
        if getattr(local_result, "fitness", 0.0) < 1e-3:
            print("icp_fails!")
            local_result = local_icp_algorithm(
                src_down, tgt_down, np.eye(4, dtype=np.float64), threshold
            )

    return local_result


def reconstruct(args):
    voxel_size = 0.08   # good balance of speed and accuracy

    rgb_dir   = os.path.join(args.data_root, "rgb")
    depth_dir = os.path.join(args.data_root, "depth")

    rgb_files   = sorted(glob.glob(os.path.join(rgb_dir,   "*.png")),
                         key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")),
                         key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    # Load GT poses
    gt_pose_path = os.path.join(args.data_root, "GT_pose.npy")
    gt_poses = _load_gt_poses(gt_pose_path)

    num_frames = min(len(rgb_files), len(depth_files), len(gt_poses))
    if num_frames == 0:
        raise RuntimeError("No frames found.")

    rgb_files   = rgb_files[:num_frames]
    depth_files = depth_files[:num_frames]
    gt_poses    = gt_poses[:num_frames]

    # ── Pre-load & pre-process all frames ──────────────────────────────
    print("Pre-loading frames...")
    raw_pcds  = []
    proc_pcds = []                       # list of (pcd_down, fpfh)
    for rgb_path, dep_path in zip(rgb_files, depth_files):
        import cv2
        rgb   = cv2.cvtColor(cv2.imread(rgb_path,   cv2.IMREAD_COLOR),    cv2.COLOR_BGR2RGB)
        depth = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)

        pcd = depth_image_to_point_cloud(rgb, depth)
        raw_pcds.append(pcd)
        proc_pcds.append(preprocess_point_cloud(pcd, voxel_size))

    # ── Initialise with GT frame-0 pose so coordinate systems align ────
    world_from_first = gt_poses[0].copy()        # 4×4
    print(f"world_from_first:\n{world_from_first}")

    # Apply world_from_first to the first raw cloud
    first_cloud = deepcopy(raw_pcds[0])
    print(f"first_cloud:\n{first_cloud}")
    first_cloud.transform(world_from_first)
    print(f"first_cloud:\n{first_cloud}")

    accumulated_pcd   = first_cloud
    camera_poses      = [world_from_first]
    first_from_prev   = np.eye(4, dtype=np.float64)

    # ── Frame-by-frame registration ────────────────────────────────────
    for i in range(1, num_frames):
        #if i % 5 == 0:
            # print(f"Processing frame {i} / {num_frames} , it has been {time.time() - start}s")

        src_down, src_fpfh = proc_pcds[i]
        tgt_down, tgt_fpfh = proc_pcds[i - 1]

        local_result = _register_pair(
            src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size, args.version
        )

        # Accumulate relative transform
        T_relative          = local_result.transformation
        print(f"trans_icp: {T_relative}")
        first_from_current  = first_from_prev @ T_relative
        print(f"first_from_current: {first_from_current}")
        world_from_current  = world_from_first @ first_from_current
        print(f"world_from_current: {world_from_current}")

        camera_poses.append(world_from_current)

        # Merge raw (full-resolution) cloud into map
        aligned = deepcopy(raw_pcds[i])
        aligned.transform(world_from_current)
        accumulated_pcd += aligned
        print(f"sourced_pcd_world: {aligned}")
        first_from_prev = first_from_current

        # Periodic downsample to keep memory bounded
        if i % 10 == 0:
            accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size)

    # ── Final downsample + ceiling removal ─────────────────────────────
    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size)
    accumulated_pcd = remove_ceiling(accumulated_pcd)

    return accumulated_pcd, camera_poses, gt_poses


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor',   type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d',
                        help='"open3d" or "my_icp"')
    args = parser.parse_args()

    args.data_root = (
        "data_collection/first_floor/"  if args.floor == 1
        else "data_collection/second_floor/"
    )

    start = time.time()
    result_pcd, pred_poses, gt_poses = reconstruct(args)
    print(f"Total execution time: {time.time() - start:.2f}s")

    visualize_and_evaluate(result_pcd, pred_poses, gt_poses, args)