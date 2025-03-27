<h1 align="center"> 3D Object Reconstruction Through Integration of Hyperspectral and RGB-D Imaging </h1>

## 📄 Abstract

This work explores the integration of RGB-D and hyperspectral imaging technologies to enhance the accuracy of 3D reconstructions. We propose a methodology that combines depth maps from an RGB-D camera with spectral data from a hyperspectral camera to develop a detailed 3D hyperspectral point cloud model. 

## 🔬 Research Methodology

### Spectral Processing Approaches
1. **HSI_Mean**: Averaging all spectral bands to create grayscale images
2. **HSI_False Color**: Extracting three specific bands (435nm, 545nm, 700nm)
3. **HSI_All**: Utilizing all available spectral bands (400-800nm with 5nm intervals)

## 🧩 Dataset

Our dataset comprises hyperspectral images of diverse objects:
- Cracker Box (rectangular)
- Chips Can (cylindrical)
- Power Drill (irregular-shaped)
- Wood Block (wooden surface)

## Prerequisites
- Python 3.8+
- OpenCV
- NumPy
- Open3D
- Required libraries: `pip install opencv-python numpy open3d`

## 🚀 Quick Start Guide
## Step 1: Feature Matching

### Running the Script
1. Navigate to the script directory
2. Run `python 1_Feature_Matching_All.py`

### Object Selection
When prompted, you'll see available objects:
```
Available objects:
1. Chips can (26 frames)
2. Cracker box (25 frames)
3. Power drill (29 frames)
4. Wood block (24 frames)
```

### Recommended First Run
- Select object: `2` (Cracker box)
- Distance threshold: `10`

### What Happens
- The script processes hyperspectral image bands
- Performs feature matching between consecutive frames
- Computes transformation matrices
- Saves transformation data in `transformation/all/Cracker box/`

### Output
- Multiple `.npz` files containing:
  - Transformation matrix
  - Fitness score
  - Number of valid matches
  - Distance threshold used

## Step 2: Point Cloud Registration

### Running the Script
1. Ensure feature matching results exist
2. Run `python 2_Registration_All.py`
3. Select the same object as in Step 1 (Cracker box)

### What Happens
- Converts depth and RGB images to point clouds
- Applies transformations from Step 1
- Performs global optimization
- Removes outliers and downsamples point clouds

### Output
The script generates a final point cloud file:
`results/all/Cracker box/3_final_result_All.ply`

![](Evaluation/Cracker%20box.gif)






## 📊 Performance Metrics

### Chamfer Distance
- Bidirectional similarity metric between point clouds
- Measures average minimum point-to-point distances

### Hausdorff Distance
- Maximum distance between point sets
- Captures worst-case point displacement

Metrics assess reconstruction accuracy across spectral processing approaches:
- HSI_Mean
- HSI_False Color
- HSI_All

## 🖥️ Visualization

Recommended Tools:
- CloudCompare
- MeshLab
- Blender

## 📄 License

[Specify license]

## 📚 Citation

If you use this work in your research, please cite:
```
[Publication Details]
```
