# GeneST

GeneST is a deep learning and transformer based spatial transcriptomics image-processing and segmentation workflow. It identifies spatially distributed genes with biological interpretable insights. The pipeline is demonsrated leveraging high-performance computing for spatial transcriptomic identification of CDX2 Genes in intestinal crypts. 



![](results/Gene_ST.png)

## Repository Layout

```text
Gene_ST/
├── src/
│   ├── pipeline/         # End-to-end training/inference pipeline scripts
│   ├── preprocessing/    # Data preprocessing and patch selection tools
│   ├── validation/       # Verification/QA scripts
│   └── visualization/    # Feature visualization scripts
├── notebooks/            # Exploratory and validation notebooks
├── plots/                # Lightweight plots/assets
├── dataset_raw/          # Raw data 
├── dataset_processed/    # Processed data 
├── dataset_yolo/         # YOLO-format dataset 
├── models/               # Trained weights/checkpoints 
├── results/              # Inference/refinement outputs 
├── runs/                 # Training runs/logs 
├── analysis/             # Large analysis artifacts 
├── requirements.txt
├── environment.yml
└── .gitignore
```

## **Features**
* **Patch Selector:** An interactive tool to load large .czi, .tif, etc. files to select patches or region of interest from the images.
* **DINOv3 Distillation Pre-Training:** Pre-training on unlabelled data using lightly train (DINOv3 as teacher and YOLOv11 backbone as student).
* **Image Segmentation Model:** YOLOv11 segmentation model with pretrain backbone weights.
* **SAM Integration:** Integrated SAM for refined mask generation.



## Setup

1. Create an environment from `environment.yml` or `requirements.txt`.
2. Install dependencies.


## Main Scripts

- `src/preprocessing/preprocess_train.py`
- `src/pipeline/1_dinov3_pretrain.py`
- `src/pipeline/2_prepare_yolo_dataset.py`
- `src/pipeline/3_train_yolo11_segmentation.py`
- `src/pipeline/4_inference_test_set.py`
- `src/pipeline/5_inference_external_patches.py`
- `src/pipeline/6_sam_refinement_inference_test.py`
- `src/pipeline/7_sam_refinement_test_set.py`



## **Acknowledgments**


1. **University of California San Diego (Boolean Lab)**
2. **Perlmutter Supercomputer**
3. **Lawrence Berkeley National Laboratory**
4. **National Energy Research Scientific Computing Center**




## **Citation**

If you use this work in your research, please cite it as follows:

```bibtex
@software{Nanda_GeneST_2026,
  author = {Amitash Nanda, H M Zabir Haque, Debashis Sahoo },
  title = {{GeneST: Identification of Spatially Distributed Gene with Biologically Interpretable Insights}},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/amitashnanda/Gene_ST}
}
```