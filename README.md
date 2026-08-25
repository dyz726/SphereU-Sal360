# SphereU-Sal360

Official PyTorch implementation of **SphereU-Sal360**, an end-to-end spherical U-shaped spatio-temporal Transformer for 360-degree video saliency prediction.

Yanzhi Ding, Kao Zhang, Zhigeng Pan, Zhihua Hu, Ming Li, and Tao Song. **SphereU-Sal360: Spherical U-Shaped Spatio-Temporal Transformer for 360° Video Saliency** (2026).

GitHub: https://github.com/dyz726/SphereU-Sal360

SphereU-Sal360 operates directly on a subdivided icosahedral mesh and avoids the polar stretching and seam discontinuities introduced by ERP/CMP projection-based learning. Its main components are:

- **Spherical U-shaped encoder-decoder:** performs multi-scale feature extraction and saliency-map reconstruction directly on an icosphere.
- **GSPE:** geometry-aware spherical positional encoding that injects global latitude-longitude orientation cues.
- **MASTA:** motion-aware spherical spatio-temporal attention that combines frame-difference-guided temporal attention with local spherical-neighborhood spatial attention.

## Results

Quantitative results reported in the paper are shown below. Higher is better for all metrics.

| Dataset | AUC-J | NSS | SIM | CC |
| --- | ---: | ---: | ---: | ---: |
| SD360 | 0.938 | 4.651 | 0.506 | 0.705 |
| AVS-ODV | 0.936 | 3.699 | 0.435 | 0.581 |
| SVGC-AVA | 0.932 | 3.597 | 0.487 | 0.665 |

## Installation

The code is implemented in Python with PyTorch. Install the required packages in your environment:

```bash
pip install torch torchvision numpy opencv-python tqdm wandb tensorboard trimesh timm einops scipy scikit-image pillow hdf5storage
```

The experiments in the paper were conducted on a single NVIDIA Tesla V100-SXM2 GPU with 16 GB memory.

## Data Preparation

The code supports `Sports-360` (SD360), `AVS-ODV`, and `SVGC_AVA`.

For Sports-360/SD360 and SVGC-AVA, arrange the data as follows:

```text
DATASET_ROOT/
├── training/
│   └── VIDEO_ID/
│       ├── maps/
│       └── fixation/
├── testing/
│   └── VIDEO_ID/
│       ├── maps/
│       └── fixation/
└── videos/
    ├── train/
    │   └── VIDEO_ID.mp4
    └── test/
        └── VIDEO_ID.mp4
```

For AVS-ODV, arrange the extracted frames and annotations as follows:

```text
DATASET_ROOT/
├── frames/VIDEO_ID/0001.jpg
├── maps/VIDEO_ID/0001_e.jpg
├── fixation/VIDEO_ID/0001_efix.png
├── train_list_1.txt
└── test_list_1.txt
```

The AVS-ODV loader supports split IDs `1`, `2`, and `3` through `--avs_split`.

## Training

The paper uses 30-frame clips, an order-6 icosphere with 40,962 vertices, an Adam optimizer, an initial learning rate of `1e-4`, a weight decay of `1e-5`, a temporal window radius of `5`, and a 2-hop spherical neighborhood. These settings are the defaults in `train.py`.

```bash
python train.py \
    --dataset_name Sports-360 \
    --dataset_root_dir /path/to/SD360 \
    --seq_length 30 \
    --img_rank 6 \
    --num_epochs 100 \
    --log_dir log/sd360
```

To fine-tune from pretrained weights, add:

```bash
--base_model_weights /path/to/pretrained_model.pth
```

For AVS-ODV, select the desired official split:

```bash
python train.py \
    --dataset_name AVS-ODV \
    --dataset_root_dir /path/to/AVS-ODV \
    --avs_split 1 \
    --log_dir log/avs_split_1
```

## Inference and Evaluation

Run inference with a trained checkpoint:

```bash
python inference.py \
    --dataset_name Sports-360 \
    --dataset_root_dir /path/to/SD360 \
    --base_model_weights /path/to/model.pth \
    --output_dir outputs \
    --method_name SphereU-Sal360 \
    --save_mat
```

Predicted saliency maps are saved as PNG files under:

```text
outputs/Results/Results_Oth/Saliency/SphereU-Sal360/saliency_png/
```

With `--save_mat`, per-video MATLAB files are also written to the neighboring `saliency_mat/` directory. The inference script reports both spherical metrics and ERP metrics.

## Output

- The prediction format can be changed in `inference.py`.
- Saliency predictions are saved as per-frame PNG files.
- Optional per-video `.mat` results can be generated with `--save_mat`.
- ERP predictions can be evaluated with the scripts under `evaluate/`.

**Results:** [Download from Baidu Netdisk](https://pan.baidu.com/s/1YAAv0h8PiTCnikVdFdSG2Q?pwd=wwd4) (extraction code: `wwd4`).

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grant Nos. 62201404 and 62572250), the Startup Foundation for Introducing Talent of NUIST (Grant No. 2024r061), and the Postgraduate Research & Practice Innovation Program of Jiangsu Province (Grant No. KYCX25_1654).

The implementation builds upon ideas from [SphereUFormer](https://github.com/KAIST-Visual-AI-Group/SphereUFormer).

## Paper & Citation

If you use SphereU-Sal360 in your research, please cite the following paper:

```bibtex
@misc{ding2026sphereusal360,
    author = {Ding, Yanzhi and Zhang, Kao and Pan, Zhigeng and Hu, Zhihua and Li, Ming and Song, Tao},
    title  = {SphereU-Sal360: Spherical U-Shaped Spatio-Temporal Transformer for 360-Degree Video Saliency},
    year   = {2026}
}
```

## Contact

Zhigeng Pan (Corresponding Author)  
Hangzhou International Innovation Institute, Beihang University, Hangzhou, China  
Email: zgpan@buaa.edu.cn

