# MoCo-Flow: Neural Motion Consensus Flow for Dynamic Humans in Stationary Monocular Cameras (Eurographics 2022)

[Project Page](https://wyysf-98.github.io/MoCo_Flow/) | [Paper](https://arxiv.org/pdf/2106.04477.pdf)

![monocular](https://wyysf-98.github.io/MoCo_Flow/assets/images/teaser.gif)

This is an official implementation. Any questions or discussions are welcomed!


## Prerequisite

### `Setup environment`
 - Python 3.8
 - PyTorch 1.9.0
 - [KNN_CUDA](https://github.com/unlimblue/KNN_CUDA)
 - [VIBE](https://github.com/mkocabas/VIBE) for human pose estimation. Please follow their installation tutorial, and move to scripts.
 - [SMPL](https://smplify.is.tue.mpg.de/). Download the SMPL models, and unpack to **utils/smpl/data**. 
 - [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) for video matting.

The final file structure shuold be as following:

```
├── scripts
│   ├── VIBE
│   └── ...
└── utils
    └── smpl
        └── data
            ├── basicmodel_f_lbs_10_207_0_v1.1.0.pkl
            ├── basicmodel_m_lbs_10_207_0_v1.1.0.pkl
            └── basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
```

Install the required packages.

    pip install -r docker/requirements.txt

** We also provide a Dockerfile for easy installation.

    docker build -t moco_flow:latest ./


## Run on [People-Snapshot](https://graphics.tu-bs.de/people-snapshot) / [ZJU-Mocap]((https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset)) / In-the-wild Monocular Data

**Please note that we do *not* use any other data, such as SMPL parameters from People-Snapshot/ZJU-Mocap. We treat all the monocular videos are in-the-wild for practicality.** 


### `Preprocess the monocular video`

** To get the best result, we highly recommend a video clip that meets the similar requirements like [Human-NeRF](https://github.com/chungyiweng/humannerf):

First, get a video and place into a folder, such as **data/youtube/videos/invisible_trap.mp4**. 

Second, run the data preprocessing script.

    VIDEO_PATH='../data/youtube/videos/invisible_trap.mp4' # input video path
    SAVE_PATH='../data/youtube/invisible_trap'             # output folder 
    START_FRAME=2295                                       # start frame of the video
    END_FRAME=2415                                         # end frame of the video
    INTERVAL=1                                             # sampling interval

    cd scripts
    python preprocess_data.py --input_video $VIDEO_PATH \
                            --output_folder $SAVE_PATH \
                            --start_frame $START_FRAME \
                            --end_frame $END_FRAME \
                            --interval $INTERVAL
    cd ..

Then you shoud get a folder in $SAVE_PATH as following:

```
└── data
    └── youtube
        └── invisible_trap
            ├── images          # images without background
            ├── images_w_bkgd   # images with background
            ├── init_nerf       # rendered images to initialize canonical NeRF
            ├── background.png  # static background image, a trick to prevent imperfect matting
            ├── train.json      # annotated file
            ├── val.json        # annotated file
            ├── xxx.mp4         # rendered video for pose estimation
            └── vibe_output.pkl # raw VIBE output file
```


Finally, modify the yaml file in configs, change the `dataloader.root_dir` in init_nerf.yaml, init_nof.yaml and c2f.yaml. We also have detailed comments in the configuration file.


### `Train models`

We use 8 GPUs (NVIDIA Tesla V100) to train the models, which takes about 3 days as mentioned in our paper.
And we will provide the pre-trained model ASAP.

First, you should initialize the canonical NeRF model.

    python train.py -c configs/youtube_invisible_trap/init_nerf.yaml

Second, for fast convergence, you can initialize the forward/backward NoF model separately. 

    python train.py -c configs/youtube_invisible_trap/init_nof.yaml

Finally, joint training using coarse to fine.
For distributed training, you should use:

    python -m torch.distributed.launch \
            --nproc_per_node=8 train.py \
            -c ./configs/youtube_invisible_trap/c2f.yaml \
            --dist

Or, for sanity check, you can use:

    python train.py -c configs/youtube_invisible_trap/c2f.yaml


### `Render output`

Render the frame input (i.e., observed motion sequence).


Run free-viewpoint rendering on a particular frame (e.g., frame 128).


Render the learned canonical appearance.

In addition, you can find the rendering scripts in `scripts/zju_mocap`.


## Acknowledgement

The implementation partly took reference from [nerf_pl](https://github.com/kwea123/nerf_pl). We thank the authors for their generosity to release code.


## Citation

If you find our work useful for your research, please consider citing using the following BibTeX entry.

```BibTeX
@article{mocoflow,
         title = {MoCo-Flow: Neural Motion Consensus Flow for Dynamic Humans in Stationary Monocular Cameras},
         author = {Xuelin Chen and Weiyu Li and Daniel Cohen-Or and Niloy J. Mitra and Baoquan Chen},
         year = {2022},
         journal = {Computer Graphics Forum},
         volume = {41},
         number = {2},
         organization = {Wiley Online Library}
}
```