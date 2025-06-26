# registration_pipeline.py
import cv2
import numpy as np
import os
import json
import copy
import open3d as o3d

def select_object(ConfigClass):
    """Select object to process using the provided Config class."""
    print("\nAvailable objects:")
    for i, (key, config) in enumerate(ConfigClass.OBJECT_CONFIGS.items(), 1):
        print(f"{i}. {config['name']} ({config['num_frames']} frames)")

    while True:
        try:
            choice = int(input(f"\nSelect object number (1-{len(ConfigClass.OBJECT_CONFIGS)}): "))
            if 1 <= choice <= len(ConfigClass.OBJECT_CONFIGS):
                return list(ConfigClass.OBJECT_CONFIGS.keys())[choice - 1]
            print(f"Invalid choice. Please select a number between 1-{len(ConfigClass.OBJECT_CONFIGS)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def remove_outliers_and_downsample(pcd, voxel_size=0.0001):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"Saved point cloud to {save_path}")

def combine_point_clouds(point_clouds, transforms):
    combined_pcd = o3d.geometry.PointCloud()
    for i, pcd in enumerate(point_clouds):
        temp_pcd = copy.deepcopy(pcd)
        temp_pcd.transform(transforms[i])
        combined_pcd += temp_pcd
    return combined_pcd

def depth_to_point_cloud(rgb, depth, mask, ConfigClass):
    h, w = depth.shape
    y, x = np.meshgrid(range(h), range(w), indexing='ij')
    valid = (mask > 0) & (depth > 0)
    z = depth[valid] / 1000.0  # Convert mm to meters
    points = np.zeros((np.sum(valid), 3))
    points[:, 0] = (x[valid] - ConfigClass.CAMERA_PARAMS['cx']) * z / ConfigClass.CAMERA_PARAMS['fx']
    points[:, 1] = (y[valid] - ConfigClass.CAMERA_PARAMS['cy']) * z / ConfigClass.CAMERA_PARAMS['fy']
    points[:, 2] = z
    colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[valid] / 255.0
    return points, colors

def get_information_matrix(source, target, transform):
    threshold = 0.001
    source.estimate_normals()
    target.estimate_normals()
    return o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, threshold, transform)

def run_registration(ConfigClass):
    """
    Main function to run the entire registration pipeline.
    Accepts a configuration class to define paths and parameters.
    """
    object_key = select_object(ConfigClass)
    object_config = ConfigClass.OBJECT_CONFIGS[object_key]
    object_name = object_config['name']
    num_images = object_config['num_frames']

    print(f"\nProcessing {object_name} with {num_images} frames using '{ConfigClass.__name__}' configuration...")
    results_dir = ConfigClass.get_results_dir(object_name)
    os.makedirs(results_dir, exist_ok=True)

    voxel_size = 0.0001
    point_clouds, point_clouds_down, initial_transforms = [], [], []

    print("Loading images and converting to point clouds...")
    for i in range(1, num_images + 1):
        rgb_path = f'Image Dataset/{object_name}/HSI/RGB/undistorted_hsi_{i}.png'
        depth_path = ConfigClass.get_depth_path(object_name, i)
        mask_path = ConfigClass.get_mask_json_path(object_name, i)
        
        rgb_img = cv2.imread(rgb_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        with open(mask_path, 'r', encoding='utf-8') as f:
            roi_points = np.array(json.load(f)['shapes'][0]['points'], dtype=np.int32)
        mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi_points], 255)

        points, colors = depth_to_point_cloud(rgb_img, depth_img, mask, ConfigClass)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        pcd_down = remove_outliers_and_downsample(pcd, voxel_size)
        point_clouds.append(pcd)
        point_clouds_down.append(pcd_down)
        initial_transforms.append(np.identity(4))

    print("\nApplying initial transformations...")
    current_transform = np.identity(4)
    for i in range(num_images - 1):
        data = np.load(ConfigClass.get_transformation_data_path(object_name, i + 1, i + 2))
        current_transform = np.matmul(current_transform, np.linalg.inv(data['transformation']))
        initial_transforms[i + 1] = current_transform
    
    print("\nCreating pose graph...")
    pose_graph = o3d.pipelines.registration.PoseGraph()
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))

    # Add sequential edges
    for i in range(num_images - 1):
        data = np.load(ConfigClass.get_transformation_data_path(object_name, i + 1, i + 2))
        transformation = data['transformation']
        information = get_information_matrix(point_clouds_down[i], point_clouds_down[i+1], transformation)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(initial_transforms[i+1]))
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, i+1, transformation, information, uncertain=False))

    # Add loop closure edge
    print("\nAdding loop closure...")
    data_last = np.load(ConfigClass.get_transformation_data_path(object_name, num_images, 1))
    transformation_last = data_last['transformation']
    information_last = get_information_matrix(point_clouds_down[-1], point_clouds_down[0], transformation_last)
    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(num_images-1, 0, transformation_last, information_last, uncertain=False))

    print("\nPerforming global optimization...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.015, edge_prune_threshold=0.25,
        preference_loop_closure=2.0, reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    final_transforms = [node.pose for node in pose_graph.nodes]
    final_combined = combine_point_clouds(point_clouds, final_transforms)
    
    print("\nPerforming final post-processing...")
    processed_pcd = remove_outliers_and_downsample(final_combined, voxel_size)

    print("\nVisualizing and saving final result...")
    final_ply_path = ConfigClass.get_final_ply_path(results_dir)
    visualize_and_save_point_cloud(processed_pcd, "Final Result", final_ply_path)
    print("\nPipeline finished successfully!")