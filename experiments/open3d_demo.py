from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / 'output' / 'experiments' / 'open3d'
RAW_PLY = OUTPUT_DIR / '01_raw_scene.ply'
DOWNSAMPLED_PLY = OUTPUT_DIR / '02_downsampled.ply'
CLEAN_PLY = OUTPUT_DIR / '03_denoised.ply'
PLANE_PLY = OUTPUT_DIR / '04_plane.ply'
OBJECTS_PLY = OUTPUT_DIR / '05_non_plane_objects.ply'
RAW_PNG = OUTPUT_DIR / '01_raw_scene.png'
CLEAN_PNG = OUTPUT_DIR / '03_denoised.png'
CLUSTER_PNG = OUTPUT_DIR / '05_clusters.png'
CSV_PATH = OUTPUT_DIR / 'cluster_summary.csv'
SEED = 42
VOXEL_SIZE = 0.025
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 1.5
PLANE_DISTANCE_THRESHOLD = 0.015
PLANE_RANSAC_N = 3
PLANE_ITERATIONS = 1000
DBSCAN_EPS = 0.06
DBSCAN_MIN_POINTS = 20

def make_synthetic_scene():
    rng = np.random.default_rng(SEED)
    plane_count = 5000
    plane_x = rng.uniform(-1.0, 1.0, plane_count)
    plane_y = rng.uniform(-1.0, 1.0, plane_count)
    plane_z = rng.normal(0.0, 0.003, plane_count)
    plane_points = np.column_stack([plane_x, plane_y, plane_z])
    object_a_count = 1200
    object_a = np.column_stack([rng.uniform(-0.55, -0.18, object_a_count), rng.uniform(-0.05, 0.28, object_a_count), rng.uniform(0.08, 0.36, object_a_count)])
    object_b_count = 1000
    object_b = np.column_stack([rng.uniform(0.25, 0.62, object_b_count), rng.uniform(-0.42, -0.12, object_b_count), rng.uniform(0.06, 0.3, object_b_count)])
    outlier_count = 200
    outliers = np.column_stack([rng.uniform(-1.5, 1.5, outlier_count), rng.uniform(-1.5, 1.5, outlier_count), rng.uniform(-0.4, 1.0, outlier_count)])
    all_points = np.vstack([plane_points, object_a, object_b, outliers])
    return all_points

def numpy_to_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def save_point_cloud_plot(points, output_path, title, labels=None):
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection='3d')
    if labels is None:
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=2)
    else:
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels, s=4)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 88)
    print('OPEN3D POINT-CLOUD DEMO')
    print('=' * 88)
    points = make_synthetic_scene()
    raw_cloud = numpy_to_point_cloud(points)
    o3d.io.write_point_cloud(str(RAW_PLY), raw_cloud)
    save_point_cloud_plot(points, RAW_PNG, '01 Raw Point Cloud')
    print()
    print(f'Raw points = {len(raw_cloud.points)}')
    downsampled_cloud = raw_cloud.voxel_down_sample(voxel_size=VOXEL_SIZE)
    o3d.io.write_point_cloud(str(DOWNSAMPLED_PLY), downsampled_cloud)
    print(f'After voxel downsample = {len(downsampled_cloud.points)}')
    clean_cloud, kept_indices = downsampled_cloud.remove_statistical_outlier(nb_neighbors=OUTLIER_NB_NEIGHBORS, std_ratio=OUTLIER_STD_RATIO)
    o3d.io.write_point_cloud(str(CLEAN_PLY), clean_cloud)
    clean_points = np.asarray(clean_cloud.points)
    save_point_cloud_plot(clean_points, CLEAN_PNG, '03 After Denoising')
    print(f'After outlier removal = {len(clean_cloud.points)}')
    removed_count = len(downsampled_cloud.points) - len(clean_cloud.points)
    print(f'Removed outlier points = {removed_count}')
    plane_model, plane_indices = clean_cloud.segment_plane(distance_threshold=PLANE_DISTANCE_THRESHOLD, ransac_n=PLANE_RANSAC_N, num_iterations=PLANE_ITERATIONS)
    a, b, c, d = plane_model
    plane_cloud = clean_cloud.select_by_index(plane_indices)
    object_cloud = clean_cloud.select_by_index(plane_indices, invert=True)
    o3d.io.write_point_cloud(str(PLANE_PLY), plane_cloud)
    o3d.io.write_point_cloud(str(OBJECTS_PLY), object_cloud)
    print()
    print('Plane model:')
    print(f'{a:.5f} x + {b:.5f} y + {c:.5f} z + {d:.5f} = 0')
    print(f'Plane points = {len(plane_cloud.points)}')
    print(f'Non-plane points = {len(object_cloud.points)}')
    labels = np.array(object_cloud.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False))
    object_points = np.asarray(object_cloud.points)
    valid_cluster_labels = sorted([int(label) for label in np.unique(labels) if label >= 0])
    print()
    print(f'DBSCAN clusters found = {len(valid_cluster_labels)}')
    rows = []
    for cluster_id in valid_cluster_labels:
        mask = labels == cluster_id
        cluster_points = object_points[mask]
        centroid = cluster_points.mean(axis=0)
        print(f'Cluster {cluster_id}: points={len(cluster_points)}, centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})')
        rows.append({'cluster_id': cluster_id, 'point_count': len(cluster_points), 'centroid_x': f'{centroid[0]:.6f}', 'centroid_y': f'{centroid[1]:.6f}', 'centroid_z': f'{centroid[2]:.6f}'})
    noise_points = int(np.sum(labels < 0))
    print(f'DBSCAN noise points = {noise_points}')
    save_point_cloud_plot(object_points, CLUSTER_PNG, '05 Non-plane DBSCAN Clusters', labels=labels)
    with CSV_PATH.open('w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['cluster_id', 'point_count', 'centroid_x', 'centroid_y', 'centroid_z'])
        writer.writeheader()
        writer.writerows(rows)
    quick_pass = len(valid_cluster_labels) >= 2
    print()
    print('=' * 88)
    print('OPEN3D CHECK')
    print('=' * 88)
    print(f'At least 2 object clusters found = {quick_pass}')
    print()
    print('Output folder:')
    print(OUTPUT_DIR)
    print()
    print('Key files:')
    print(RAW_PNG)
    print(CLEAN_PNG)
    print(CLUSTER_PNG)
    print(CSV_PATH)
    print()
    print('This is a point-cloud processing demo, not a SLAM system.')
if __name__ == '__main__':
    main()
