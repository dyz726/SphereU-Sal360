# SphereU-Sal360

It is the official implementation code for the SphereU-Sal360 model.
* Yanzhi Ding, Kao Zhang, Zhigeng Pan, Zhihua Hu, Ming Li, and Tao Song. SphereU-Sal360: Spherical U-Shaped Spatio-Temporal Transformer for 360° Video Saliency. KSEM 2026.

Github: https://github.com/dyz726/SphereU-Sal360

## Dataset Preparation

For Sports-360 and SVGC-AVA, organize the dataset as follows:

```text
DATASET_ROOT/
├── training/
│   └── VIDEO_ID/
│       ├── maps/
│       │   └── FRAME_ID.png
│       └── fixation/
│           └── FRAME_ID.png
├── testing/
│   └── VIDEO_ID/
│       ├── maps/
│       │   └── FRAME_ID.png
│       └── fixation/
│           └── FRAME_ID.png
└── videos/
    ├── train/
    │   └── VIDEO_ID.mp4
    └── test/
        └── VIDEO_ID.mp4
```

For AVS-ODV, organize the dataset as follows:

```text
DATASET_ROOT/
├── frames/
│   └── VIDEO_ID/
│       └── 0001.jpg
├── maps/
│   └── VIDEO_ID/
│       └── 0001_e.jpg
├── fixation/
│   └── VIDEO_ID/
│       └── 0001_efix.png
├── train_list_1.txt
├── test_list_1.txt
├── train_list_2.txt
├── test_list_2.txt
├── train_list_3.txt
└── test_list_3.txt
```

## Training

Sports-360:

```bash
python train.py \
    --dataset_name Sports-360 \
    --dataset_root_dir /path/to/Sports-360 \
    --seq_length 30 \
    --img_rank 6 \
    --num_epochs 100 \
    --log_dir log/sports360
```

SVGC-AVA:

```bash
python train.py \
    --dataset_name SVGC_AVA \
    --dataset_root_dir /path/to/SVGC_AVA \
    --seq_length 30 \
    --img_rank 6 \
    --num_epochs 100 \
    --log_dir log/svgc_ava
```

AVS-ODV (use `--avs_split` to select split 1, 2, or 3):

```bash
python train.py \
    --dataset_name AVS-ODV \
    --dataset_root_dir /path/to/AVS-ODV \
    --avs_split 1 \
    --seq_length 30 \
    --img_rank 6 \
    --num_epochs 100 \
    --log_dir log/avs_odv_split_1
```

To fine-tune from pretrained weights, add:

```bash
--base_model_weights /path/to/pretrained_model.pth
```

## Inference

```bash
python inference.py \
    --dataset_name Sports-360 \
    --dataset_root_dir /path/to/Sports-360 \
    --base_model_weights /path/to/model.pth \
    --output_dir outputs \
    --method_name SphereU-Sal360 \
    --seq_length 30 \
    --img_rank 6 \
    --save_mat
```

For AVS-ODV inference, change `--dataset_name` and `--dataset_root_dir`, and add the required split:

```bash
--dataset_name AVS-ODV --dataset_root_dir /path/to/AVS-ODV --avs_split 1
```

## Output
And it is easy to change the output format in our code.
* The results of video task are saved in ".png" and optional ".mat" formats.
* The output package also contains the trained model weights (`.pth`). These checkpoints can be used directly for inference by passing the selected file to `--base_model_weights`.
* You can evaluate the performance based on the scripts in the "evaluate" folder.

**Results**: [ALL](https://pan.baidu.com/s/1YAAv0h8PiTCnikVdFdSG2Q?pwd=wwd4) (extraction code: wwd4)

# Acknowledgments
This research was funded by: the National Natural Science Foundation of China (Grant Nos. 62201404 and 62572250), the Startup Foundation for Introducing Talent of NUIST (Grant No. 2024r061), and the Postgraduate Research & Practice Innovation Program of Jiangsu Province (Grant No. KYCX25_1654).


## Paper & Citation

If you use the SphereU-Sal360 360° video saliency model, please cite the following paper:
```
@InProceedings{Ding_2026_KSEM,
    author    = {Ding, Yanzhi and Zhang, Kao and Pan, Zhigeng and Hu, Zhihua and Li, Ming and Song, Tao},
    title     = {SphereU-Sal360: Spherical U-Shaped Spatio-Temporal Transformer for 360-Degree Video Saliency},
    booktitle = {International Conference on Knowledge Science, Engineering and Management (KSEM)},
    year      = {2026}
}
```


## Contact
Kao ZHANG  <br />
3D Reconstruction and Image Processing Group (3DIP) <br />
Perceptual and Generative AI Lab (PGAI Lab) <br />
Nanjing University of Information Science and Technology, Nanjing, China. <br />
Email: kaozhang@nuist.edu.cn <br />

Yanzhi Ding  <br />
3D Reconstruction and Image Processing Group (3DIP) <br />
Perceptual and Generative AI Lab (PGAI Lab) <br />
Nanjing University of Information Science and Technology, Nanjing, China. <br />
Email: yanzhiding@nuist.edu.cn <br />
