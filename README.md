# SphereU-Sal360

It is the official implementation code for the SphereU-Sal360 model.

* Yanzhi Ding, Kao Zhang, Zhigeng Pan, Zhihua Hu, Ming Li, and Tao Song. SphereU-Sal360: Spherical U-Shaped Spatio-Temporal Transformer for 360° Video Saliency. KSEM 2026.

Github: https://github.com/dyz726/SphereU-Sal360

## Dataset Preparation

Organize the datasets as follows (one video and one frame are shown per dataset):

```text
Dataset/
├── AVS-ODV/
│   ├── frames/
│   │   └── 0_01.mp4/
│   │       └── 0001.jpg
│   ├── maps/
│   │   └── 0_01.mp4/
│   │       └── 0001_e.jpg
│   ├── fixation/
│   │   └── 0_01.mp4/
│   │       └── 0001_efix.png
│   ├── train_list_1.txt
│   └── test_list_1.txt
├── SVGC_AVA/
│   ├── videos/
│   │   └── L01.mp4
│   ├── maps/
│   │   └── L01.mp4/
│   │       └── 0.png
│   ├── fixation/
│   │   └── L01.mp4/
│   │       └── 0.png
│   ├── train_list_1.txt
│   └── test_list_1.txt
└── SD360/
    ├── videos/
    │   └── 217.mp4
    ├── maps/
    │   └── 217.mp4/
    │       └── 0001.png
    ├── fixation/
    │   └── 217.mp4/
    │       └── 0001.png
    ├── train_list_1.txt
    └── test_list_1.txt
```

Set `--dataset_root_dir` to the individual dataset directory, such as `/path/to/Dataset/SVGC_AVA`. Per-video directories retain the `.mp4` suffix: for example, `maps/L01.mp4/` and `fixation/L01.mp4/` correspond to `videos/L01.mp4`. AVS-ODV stores extracted RGB frames in `frames/`.

The split files are provided in [`Dataset_split/`](Dataset_split/). Copy the files for each dataset into its dataset root before running inference.

The datasets use the following video-level splits:

* **AVS-ODV** ([GitHub](https://github.com/IntMeGroup/AVS-ODV)): We use official **split 1**, with **132 training videos** in `train_list_1.txt` and **30 test videos** in `test_list_1.txt` (**test:train = 30:132**).
* **SVGC_AVA** ([GitHub](https://github.com/QinYang12/SVGC-AVA?tab=readme-ov-file)): Following the official random split protocol, the 57 videos are divided into **46 training videos and 11 test videos** (**train:test = 46:11**). The selected partition is recorded in `train_list_1.txt` and `test_list_1.txt`.
* **SD360** ([GitHub](https://github.com/xuyanyu-shh/Saliency-detection-in-360-video)): Following the official split protocol, `train_list_1.txt` and `test_list_1.txt` contain **80 training videos and 24 test videos**, respectively.

## Inference

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

For SD360:

```bash
python inference.py \
    --dataset_name SD360 \
    --dataset_root_dir /path/to/Dataset/SD360 \
    --base_model_weights /path/to/model.pth \
    --output_dir outputs \
    --method_name SphereU-Sal360 \
    --seq_length 30 \
    --img_rank 6 \
    --save_mat
```

For AVS-ODV inference, use official split 1:

```bash
--dataset_name AVS-ODV --dataset_root_dir /path/to/Dataset/AVS-ODV --dataset_split 1
```

## Output

* Predictions are saved as PNG images. Use `--save_mat` to also export per-video MAT files.
* Model weights (`.pth`) are included in the results package and can be passed to `--base_model_weights` for inference.
* Evaluation scripts are provided in `evaluate/`.

**Weights and Results**: [ALL](https://pan.baidu.com/s/1TLQsvUhV4dnufWelAtysBw?pwd=qgbi) (extraction code: qgbi)

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
Yanzhi Ding  <br />
Nanjing University of Information Science and Technology, Nanjing, China. <br />
Email: yanzhiding@nuist.edu.cn <br />
