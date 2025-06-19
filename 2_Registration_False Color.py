import cv2
import numpy as np
import os
import json
import copy
import open3d as o3d
from matplotlib import pyplot as plt
from config import CommonConfig # Import CommonConfig

# Define configuration specific to this script
class Config(CommonConfig):
    """Configuration parameters management class for 'False Color' registration."""
    
    # Path for loading transformation data from the first stage
    @staticmethod
    def get_transformation_data_path(object_name, i, j):
        return f'transformation/False Color/{object_name}/transformation_{i}_{j}.npz'

    # Output directory for registration results
    @staticmethod
    def get_results_dir(object_name):
        return f'results/False Color/{object_name}'

    # Output PLY file path for final result
    @staticmethod
    def get_final_ply_path(results_dir):
        return f"{results_dir}/3_final_result_False Color.ply"

# No need to redefine fx, fy, cx, cy, OBJECT_CONFIGS here, they come from Config (inheriting CommonConfig)

def select_object():
    """Select object to process"""
    print("\nAvailable objects:")
    # Access OBJECT_CONFIGS from the inherited Config class
    for i, (key, config) in enumerate(Config.OBJECT_CONFIGS.items(), 1):
        print(f"{i}. {config['name']} ({config['num_frames']} frames)")

    while True:
        try:
            choice = int(input("\nSelect object number (1-4): "))
            if 1 <= choice <= 4:
                return list(Config.OBJECT_CONFIGS.keys())[choice - 1]
            print("Invalid choice. Please select a number between 1-4.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def remove_outliers_and_downsample(pcd, voxel_size=0.0001):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd_down.remove_statistical_outlier(
        nb_neighbors=50,
        std_ratio=1.0
    )
    return cl

def visualize_and_save_point_cloud(pcd, window_name="Point Cloud", save_path=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 1.5

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(axis)

    vis.run()
    vis.destroy_window()

    if save_path:
        output_dir = os.path.dirname(save_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Saved point cloud to {save_path}")

def combine_point_clouds(point_clouds, transforms):
    combined_pcd = o3d.geometry.PointCloud()
    for i, pcd in enumerate(point_clouds):
        temp_pcd = copy.deepcopy(pcd)
        temp_pcd.transform(transforms[i])
        combined_pcd += temp_pcd
    return combined_pcd

def depth_to_point_cloud(rgb, depth, mask):
    h, w = depth.shape
    y, x = np.meshgrid(range(h), range(w), indexing='ij')

    valid = (mask > 0) & (depth > 0)
    y = y[valid]
    x = x[valid]
    z = depth[valid] / 1000.0 # Convert from mm to meters

    points = np.zeros((len(x), 3))
    # Access camera parameters from Config
    points[:, 0] = (x - Config.CAMERA_PARAMS['cx']) * z / Config.CAMERA_PARAMS['fx']
    points[:, 1] = (y - Config.CAMERA_PARAMS['cy']) * z / Config.CAMERA_PARAMS['fy']
    points[:, 2] = z

    rgb_converted = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    colors = rgb_converted[valid] / 255.0

    return points, colors

def load_edge_info(object_name, i, j):
    # Use the specific path defined in Config for this script type
    data = np.load(Config.get_transformation_data_path(object_name, i, j))
    return {
        'source': i,
        'target': j,
        'transformation': data['transformation'],
        'fitness': data['fitness'],
        'valid_matches_count': data['valid_matches_count']
    }

def get_information_matrix(source, target, transform):
    threshold = 0.001
    source.estimate_normals()
    target.estimate_normals()
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, threshold, transform)
    return information

def main():
    # Select object to process
    object_key = select_object()
    # Access OBJECT_CONFIGS from the inherited Config class
    object_config = Config.OBJECT_CONFIGS[object_key]
    object_name = object_config['name']
    num_images = object_config['num_frames']

    print(f"\nProcessing {object_name} with {num_images} frames...")

    # Create output directory for results
    results_dir = Config.get_results_dir(object_name)
    os.makedirs(results_dir, exist_ok=True)

    voxel_size = 0.0001
    point_clouds = []
    point_clouds_down = []
    initial_transforms = []
    final_transforms = []

    print("Loading images and converting to point clouds...")
    for i in range(1, num_images + 1):
        rgb_path = f'Image Dataset/{object_name}/HSI/RGB/undistorted_hsi_{i}.png'
        # Use common depth and mask paths from Config
        depth_path = Config.get_depth_path(object_name, i)
        mask_path = Config.get_mask_json_path(object_name, i)

        rgb_img = cv2.imread(rgb_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        with open(mask_path, 'r', encoding='utf-8') as f:
            mask_data = json.load(f)
        roi_points = np.array(mask_data['shapes'][0]['points'], dtype=np.int32)
        mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi_points], 255)

        points, colors = depth_to_point_cloud(rgb_img, depth_img, mask)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down = remove_outliers_and_downsample(pcd_down, voxel_size)

        point_clouds.append(pcd)
        point_clouds_down.append(pcd_down)
        initial_transforms.append(np.identity(4))

    # 2. Applying initial transformations
    print("\nApplying initial transformations...")
    current_transform = np.identity(4)
    initial_transforms[0] = current_transform

    for i in range(num_images - 1):
        edge_info = load_edge_info(object_name, i + 1, i + 2)
        # edge_info['translation'] = i+1_T_i
        current_transform = np.matmul(current_transform, np.linalg.inv(edge_info['transformation']))
        # initial_transforms = 0_T_i
        initial_transforms[i + 1] = current_transform

    initial_combined = combine_point_clouds(point_clouds_down, initial_transforms)
    initial_combined = remove_outliers_and_downsample(initial_combined, voxel_size)
    print("\nVisualizing and saving initial alignment...")
    # Uncomment the line below to visualize and save initial alignment
    # visualize_and_save_point_cloud(initial_combined, "Initial Alignment",
    #                              f"{results_dir}/1_initial_alignment_False Color.ply")

    # Create pose graph using initial transformations
    print("\nCreating pose graph...")
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for source_id in range(num_images - 1):
        target_id = source_id + 1
        print(f"Processing pair {source_id + 1} -> {target_id + 1}")

        # Use initial transformation directly instead of ICP
        transformation = load_edge_info(object_name, source_id+1, target_id+1)['transformation']
        information = get_information_matrix(
            point_clouds_down[source_id],
            point_clouds_down[target_id],
            transformation)

        # transformation for node: 0_T_i
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                initial_transforms[target_id]))

        # transformation for edge: i+1_T_i
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_id, target_id,
                transformation,
                information,
                uncertain=False))

    # Adding loop closure
    print("\nAdding loop closure...")
    edge_info_last = load_edge_info(object_name, num_images, 1)
    transformation_last = edge_info_last['transformation']
    information_last = get_information_matrix(
        point_clouds_down[-1],
        point_clouds_down[0],
        transformation_last)

    pose_graph.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(
            num_images-1, 0,
            transformation_last,
            information_last,
            uncertain=False))

    # Performing global optimization
    print("\nPerforming global optimization...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.015,
        edge_prune_threshold=0.25,
        preference_loop_closure=2.0,
        reference_node=0)

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    # Get final transforms and combine point clouds
    for i in range(num_images):
        final_transforms.append(pose_graph.nodes[i].pose)

    final_combined = combine_point_clouds(point_clouds_down, final_transforms)
    final_combined = remove_outliers_and_downsample(final_combined, voxel_size)
    print("\nVisualizing and saving global optimization result...")
    # Uncomment the line below to visualize and save global optimization result
    # visualize_and_save_point_cloud(
    #     final_combined,
    #     "After Global Optimization (No Initial ICP)",
    #     f"{results_dir}/2_after_global_optimization_False Color.ply"
    # )

    # Final post-processing
    print("\nPerforming final post-processing...")
    print("- Additional voxel downsampling...")
    processed_pcd = final_combined.voxel_down_sample(voxel_size=voxel_size)

    print("- Additional statistical outlier removal...")
    cl, ind = processed_pcd.remove_statistical_outlier(
        nb_neighbors=50,
        std_ratio=1.0
    )
    processed_pcd = cl

    print("\nVisualizing and saving final result...")
    # Use the specific path defined in Config for this script type
    visualize_and_save_point_cloud(
        processed_pcd,
        "Final Result (No Initial ICP)",
        Config.get_final_ply_path(results_dir)
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
