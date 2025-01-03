# GLIMS: Attention-guided lightweight multi-scale hybrid network for volumetric semantic segmentation

This repository contains the code of GLIMS. <br /><br /> **GLIMS** ranked in the **top 5 among 65 unique submissions** during the validation phase of the Adult Glioblastoma Segmentation challenge of BraTS 2023.

## Installation

### Clone the repository

```
git clone https://github.com/yaziciz/GLIMS.git
cd GLIMS
```

### Install the required dependencies

With your virtual environment activated, install the project's dependencies:
```
pip install -r requirements.txt
```

## Usage Instructions

### Running the Main Script

The GLIMS model can be trained by the given script on the [BraTS 2023](https://www.synapse.org/#!Synapse:syn51156910/wiki/) dataset: 

```bash
python main.py --output_dir <output_directory> --data_dir <data_directory> --json_list <json_list_file> --fold <fold_id>
```

## Validation
By using the pre-trained model, the validation phase can be performed as follows:

```bash
python post_validation.py --output_dir <output_directory> --data_dir <data_directory> --json_list <json_list_file> --fold <fold_number> --pretrained_dir <pretrained_model_directory>
```
## Testing with Model Ensembles
To test GLIMS by using the ensemble method on the unannotated BraTS 2023 dataset, the following script can be used: 

```bash
python test_BraTS.py --data_dir <validation_data_directory> --model_ensemble_1 <model_1_path> --model_ensemble_2 <model_2_path> --output_dir <output_directory>
```

The `model_ensemble_1` and `model_ensemble_2` variables represent the `fold 2` and `fold 4` models, as indicated in our challenge submission paper on arXiv.

## Citations

**GLIMS: Attention-guided lightweight multi-scale hybrid network for volumetric semantic segmentation** <br />
Image and Vision Computing, May 2024 <br />
[Journal Paper](https://www.sciencedirect.com/science/article/pii/S0262885624001598), [arXiv](https://arxiv.org/abs/2404.17854) <br /><br />

```bash
@article{yazici2024glims,
  title={GLIMS: Attention-guided lightweight multi-scale hybrid network for volumetric semantic segmentation},
  author={Yaz{\i}c{\i}, Ziya Ata and {\"O}ks{\"u}z, {\.I}lkay and Ekenel, Haz{\i}m Kemal},
  journal={Image and Vision Computing},
  pages={105055},
  year={2024},
  publisher={Elsevier},
  doi={https://doi.org/10.1016/j.imavis.2024.105055}
}
```

**Attention-Enhanced Hybrid Feature Aggregation Network for 3D Brain Tumor Segmentation**<br />
Accepted to the 9th Brain Lesion (BrainLes) Workshop @ MICCAI 2023 <br />
[Challenge Proceedings Paper](https://link.springer.com/chapter/10.1007/978-3-031-76163-8_9) [arXiv](https://arxiv.org/abs/2403.09942) <br /><br />

```bash
@incollection{yazici2023attention,
  title={Attention-Enhanced Hybrid Feature Aggregation Network for 3D Brain Tumor Segmentation},
  author={Yaz{\i}c{\i}, Ziya Ata and {\"O}ks{\"u}z, {\.I}lkay and Ekenel, Haz{\i}m Kemal},
  booktitle={International Challenge on Cross-Modality Domain Adaptation for Medical Image Segmentation},
  pages={94--105},
  year={2023},
  publisher={Springer},
  doi={https://doi.org/10.1007/978-3-031-76163-8_9}
}
```

Thank you for your interest in our work!

We are also deeply grateful to the MONAI Consortium for their [MONAI](https://arxiv.org/abs/2211.02701) framework, which was instrumental in the development of GLIMS.

