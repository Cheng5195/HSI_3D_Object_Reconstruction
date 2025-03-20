import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

def load_point_cloud(file_path):
    """Load point cloud from PLY file"""
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def scale_point_cloud(pcd, scale_factor):
    """Scale point cloud by a given factor"""
    scaled_pcd = copy.deepcopy(pcd)
    points = np.asarray(scaled_pcd.points)
    scaled_points = points * scale_factor
    scaled_pcd.points = o3d.utility.Vector3dVector(scaled_points)
    return scaled_pcd

def visualize_point_cloud(pcd_list, title="Point Cloud Visualization"):
    """Visualize one or multiple point clouds"""
    o3d.visualization.draw_geometries(pcd_list, window_name=title)

def align_point_clouds(source_pcd, target_pcd, max_distance=0.5):
    """
    Align source point cloud to target point cloud using ICP
    """
    source = copy.deepcopy(source_pcd)
    target = copy.deepcopy(target_pcd)
    
    source_center = source.get_center()
    target_center = target.get_center()
    
    initial_transform = np.eye(4)
    initial_transform[:3, 3] = target_center - source_center
    
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=max_distance,
        init=initial_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    
    source.transform(result_icp.transformation)
    return source, result_icp.transformation

def compute_chamfer_distance(source_points, target_points, chunk_size=50000):
    """
    Compute bidirectional Chamfer distance between two point clouds
    """
    def compute_directed_distance(source, target):
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
        nn.fit(target)
        
        distances = []
        for i in range(0, len(source), chunk_size):
            chunk = source[i:i+chunk_size]
            dist, _ = nn.kneighbors(chunk)
            distances.append(dist)
        
        return np.concatenate(distances).flatten()

    source_to_target = compute_directed_distance(source_points, target_points)
    target_to_source = compute_directed_distance(target_points, source_points)
    
    metrics = {
        'mean_chamfer': np.mean([np.mean(source_to_target), np.mean(target_to_source)]),
        'max_chamfer': np.max([np.max(source_to_target), np.max(target_to_source)]),
        'source_to_target': source_to_target,
        'target_to_source': target_to_source
    }
    
    return metrics

def compute_hausdorff_distance(source_points, target_points):
    nn_source = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
    nn_target = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
    
    nn_source.fit(target_points)
    nn_target.fit(source_points)
    
    source_to_target, _ = nn_source.kneighbors(source_points)
    target_to_source, _ = nn_target.kneighbors(target_points)
    
    metrics = {
        'hausdorff': max(np.max(source_to_target), np.max(target_to_source)),
        'source_to_target_max': np.max(source_to_target),
        'target_to_source_max': np.max(target_to_source)
    }
    
    return metrics

def compute_surface_smoothness(source_pcd, k_neighbors=30):
    points = np.asarray(source_pcd.points)
    
    nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto', n_jobs=-1)
    nn.fit(points)
    
    distances, indices = nn.kneighbors(points)
    smoothness_values = []
    
    for idx, neighbors_idx in enumerate(indices):
        try:
            neighbor_points = points[neighbors_idx]
            centroid = np.mean(neighbor_points, axis=0)
            neighbor_points_centered = neighbor_points - centroid
            
            _, s, _ = np.linalg.svd(neighbor_points_centered)
            
            if sum(s) != 0:
                smoothness = s[-1] / sum(s)
            else:
                smoothness = 1.0
                
            smoothness_values.append(smoothness)
        except:
            smoothness_values.append(1.0)
    
    smoothness_array = np.array(smoothness_values)
    
    metrics = {
        'mean_smoothness': np.mean(smoothness_array),
        'median_smoothness': np.median(smoothness_array),
        'max_smoothness': np.max(smoothness_array),
        'std_smoothness': np.std(smoothness_array)
    }
    
    return metrics

def color_point_cloud_by_metric(pcd, values, colormap='viridis'):
    """Color point cloud based on metric values"""
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
    colors = plt.cm.get_cmap(colormap)(normalized_values)[:, :3]
    colored_pcd = copy.deepcopy(pcd)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    return colored_pcd

def main():
    # Load point clouds
    print("Loading point clouds...")
    truth_pcd = load_point_cloud('GT/Wood block.ply')
    generated_pcd = load_point_cloud('3_final_result_all_False Color.ply')
    
    print(f"Reference cloud: {len(truth_pcd.points)} points")
    print(f"Generated cloud: {len(generated_pcd.points)} points")
    
    target_points = len(generated_pcd.points) // 8
    print(f"Target points after downsampling: {target_points}")
    
    voxel_size_min = 0.001
    voxel_size_max = 0.1
    max_iterations = 10
    best_voxel_size = None
    best_point_count = 0
    
    print("\nFinding optimal voxel size for downsampling...")
    for i in range(max_iterations):
        voxel_size = (voxel_size_min + voxel_size_max) / 2
        downsampled = generated_pcd.voxel_down_sample(voxel_size=voxel_size)
        current_points = len(downsampled.points)
        
        if current_points > target_points:
            voxel_size_min = voxel_size
        else:
            voxel_size_max = voxel_size
            
        if best_point_count == 0 or abs(current_points - target_points) < abs(best_point_count - target_points):
            best_voxel_size = voxel_size
            best_point_count = current_points
        
        if abs(current_points - target_points) < target_points * 0.1:  # Allow 10% error
            break
    
    # voxel_size
    downsampled_generated = generated_pcd.voxel_down_sample(voxel_size=best_voxel_size)
    
    if len(downsampled_generated.points) < target_points * 0.5:
        print("\nWARNING: Downsampled point count is much lower than expected!")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Operation cancelled by user.")
            return
    
    scale_factor = 5.0
    scaled_generated_pcd = scale_point_cloud(downsampled_generated, scale_factor)

    # Align point clouds
    print("\nAligning point clouds...")
    aligned_generated_pcd, transformation = align_point_clouds(scaled_generated_pcd, truth_pcd)
    
    # visualize
    print("\nVisualizing aligned point clouds...")
    visualize_point_cloud([truth_pcd, aligned_generated_pcd], "Aligned Point Clouds")
    
    print("\nComputing quality metrics...")
    truth_points = np.asarray(truth_pcd.points)
    generated_points = np.asarray(aligned_generated_pcd.points)
    
    # 1. Chamfer Distance
    print("\n=== Chamfer Distance Metrics ===")
    chamfer_metrics = compute_chamfer_distance(truth_points, generated_points)
    print(f"Mean Chamfer distance: {chamfer_metrics['mean_chamfer'] * 100:.6f} mm")
    
    # 2. Hausdorff Distance
    print("\n=== Hausdorff Distance Metrics ===")
    hausdorff_metrics = compute_hausdorff_distance(truth_points, generated_points)
    print(f"Hausdorff distance: {hausdorff_metrics['hausdorff'] * 100:.6f} mm")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()