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

def visualize_point_cloud(pcd_list, title="Point Cloud Visualization"):
    """Visualize one or multiple point clouds"""
    o3d.visualization.draw_geometries(pcd_list, window_name=title)

def align_point_clouds(source_pcd, target_pcd, max_distance=0.5):
    """
    Align source point cloud to target point cloud using ICP
    """
    source = copy.deepcopy(source_pcd)
    target = copy.deepcopy(target_pcd)
    
    # Estimate rough alignment using centroids
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
    """
    Compute bidirectional Hausdorff distance between two point clouds
    """
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

def color_point_cloud_by_metric(pcd, values, colormap='viridis'):
    """Color point cloud based on metric values"""
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
    colors = plt.cm.get_cmap(colormap)(normalized_values)[:, :3]
    colored_pcd = copy.deepcopy(pcd)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    return colored_pcd

def evaluate_point_cloud(generated_pcd, truth_pcd, name=""):
    # Align point clouds
    aligned_generated_pcd, transformation = align_point_clouds(generated_pcd, truth_pcd)
    
    truth_points = np.asarray(truth_pcd.points)
    generated_points = np.asarray(aligned_generated_pcd.points)
    
    # Calculate Chamfer distance
    chamfer_metrics = compute_chamfer_distance(truth_points, generated_points)
    
    # Calculate Hausdorff distance
    hausdorff_metrics = compute_hausdorff_distance(truth_points, generated_points)
    
    return {
        'name': name,
        'aligned_pcd': aligned_generated_pcd,
        'chamfer_metrics': chamfer_metrics,
        'hausdorff_metrics': hausdorff_metrics,
        'num_points': len(generated_points)
    }

def main():
    # Load ground truth point cloud
    print("Loading reference point cloud...")
    truth_pcd = load_point_cloud('../Image Dataset/Evaluation/GT/Cracker box.ply')
    print(f"Reference cloud: {len(truth_pcd.points)} points")
    
    # Generate list of point cloud files
    generated_files = [
        '../Image Dataset/Evaluation/Mean/Cracker box/3_final_result_Mean.ply',
        '../Image Dataset/Evaluation/False Color/Cracker box/3_final_result_False Color.ply',
        '../Image Dataset/Evaluation/All/Cracker box/3_final_result_all.ply',
    ]
    
    results = []
    
    # Process each generated point cloud
    for file_path in generated_files:
        print(f"\nProcessing {file_path}...")
        generated_pcd = load_point_cloud(file_path)
        print(f"Generated cloud: {len(generated_pcd.points)} points")
        
        # Evaluate point cloud
        result = evaluate_point_cloud(generated_pcd, truth_pcd, file_path)
        results.append(result)
        
        # Visualize current point cloud vs ground truth
        print("\nVisualizing current point cloud...")
        # visualize_point_cloud([truth_pcd, result['aligned_pcd']], f"Aligned Point Clouds - {file_path}")

    print("\n=== Comparison Results ===")
    # print("\nChamfer Distance Metrics:")
    for result in results:
        print(f"\n{result['name']}:")
        print(f"Number of points: {result['num_points']}")
        print(f"Mean Chamfer distance: {result['chamfer_metrics']['mean_chamfer']:.6f}")
        print(f"Hausdorff distance: {result['hausdorff_metrics']['hausdorff']:.6f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
