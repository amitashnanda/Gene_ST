# GeneST: A Deep Learning Framework to Identify Spatially Distributed Genes for Translational Health Impact 


---

##  `preprocessing/`

- **`color_normalize_pipe.py`**  
  Applies color normalization to patches using a reference image.

- **`GeneST_patch_selector.py`**  
  Interactive OpenCV-based GUI tool to manually select patches from large WSIs and save patch coordinates and patch images.

- **`hough_transform.py`**  
  Performs Hough transform to estimate and correct tissue orientation in patches and resizes with black padding to 1024×768.

- **`raw_image_resize.py`**  
Downscales and rescale to 1024×768 using aspect ratio preserving resizing with black padding for visualization and patch selection.

---

## `spatial_transcriptomics_testing/`

- **`preprocess_spatial_data.ipynb`**  

After extracting crypts using segmentation model, this analysis performed to identify the gene expression existence in ST images.

- **`preprocess_spatial_SI_A.ipynb`**  
  Preprocessing pipeline specific to small intestine sample A (SI_A).

- **`preprocess_spatial_SI_B.ipynb`**  
  Preprocessing pipeline specific to small intestine sample B (SI_B).

- **`preprocess_spatial_SI_C.ipynb`**  
  Preprocessing pipeline specific to small intestine sample C (SI_C).

- **`coordinates_calculation.ipynb`**  
  Utilities for converting patch coordinates between original, downsampled, and resized space. Useful for mapping predictions back to WSI coordinates.

- **`model.ipynb`**  
  Main inference script that loads the trained model and applies it to preprocessed patches for predictions.

- **`sanity_check.ipynb`**  
  Utility notebook to verify preprocessing steps, coordinate mappings, or model predictions visually and quantitatively.

---



---

## Requirements

conda env create -f environment.yml
