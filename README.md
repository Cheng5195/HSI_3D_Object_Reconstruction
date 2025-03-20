Repository Structure

Evaluation

The Evaluation folder contains code for quantitative evaluation of the different objects in our dataset.

The evaluation includes:
- 1_Feature_Matching_All: Feature matching results using all spectral bands (HSI_All)
- 1_Feature_Matching_False Color: Feature matching results using three specific bands (435nm, 545nm, 700nm) to create RGB representation
- 1_Feature_Matching_Mean: Feature matching results using grayscale images created by averaging all spectral bands
- 2_Registration_All: Registration results using all spectral bands (HSI_All)
- 2_Registration_False Color: Registration results using three specific bands (RGB representation)
- 2_Registration_Mean: Registration results using grayscale images

Dataset

Our dataset contains hyperspectral images (400-800nm range with 5nm intervals) of various objects including a rectangular-shaped Cracker box, a cylindrical Chips can, an irregular-shaped Power drill, and a wooden-surfaced Wood block for comprehensive evaluation.

Pipeline

Our pipeline implements a 3D reconstruction approach using hyperspectral imaging data processed in three different ways:
1. HSI_Mean: Averaging all spectral bands to create grayscale images
2. HSI_False Color: Extracting three bands (435nm, 545nm, 700nm) according to CIE standards
3. HSI_All: Utilizing all available spectral bands
