# 3d_pose_baseline_pytorch

A PyTorch implementation of a simple baseline for 3d human pose estimation. The orginal repository is by [3d_pose_baseline_pytorch](https://github.com/weigq/3d_pose_baseline_pytorch) 4 years ago and I made little changes in 2022. 
You can check the original Tensorflow implementation written by [Julieta Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline).




## Installation

1. First, clone this repository:
    ```
    git clone --recursive https://github.com/weigq/3d_pose_baseline_pytorch.git
    ```
2. Download the pre-processed [Human3.6M](https://drive.google.com/drive/folders/1h4S3vmtjso_rxP6_Lfg6a1ue7pJMuGp9) dataset in 3d joints:
    ```
    unzip human36m.zip
    rm h36m.zip
    ```

## Usage


### Train

1. Train on Human3.6M groundtruth 2d joints:
    ```
    # optional arguments, you can access more details in opt.py
    main.py [-h] [--data_dir DATA_DIR] [--exp EXP] [--ckpt CKPT]
               [--load LOAD] [--test] [--resume]
               [--action {all,All}]
               [--max_norm] [--linear_size LINEAR_SIZE]
               [--num_stage NUM_STAGE] [--use_hg] [--lr LR]
               [--lr_decay LR_DECAY] [--lr_gamma LR_GAMMA] [--epochs EPOCHS]
               [--dropout DROPOUT] [--train_batch TRAIN_BATCH]
               [--test_batch TEST_BATCH] [--job JOB] [--no_max] [--max]
               [--procrustes]
    ```
    train the model:
    ```
    python main.py --exp example
    ```

   
### Test

1. You can download the [pretrained model](https://drive.google.com/file/d/1-HUPATPgQJWcitsPoRAECu7hssVqXofq/view?usp=sharing) on ground-truth 2d pose for a quick demo.

    ```
    python main.py --load $PATH_TO_gt_ckpt_best.pth.tar --test
    ```
    and you will get the results:

    |  | direct. | discuss. | eat. | greet. | phone | photo | pose | purch. | sit | sitd. | somke | wait | walkd. | walk | walkT | avg |
    | :--: | :--: | :--: | :--: | :--: |  :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
    | original version | 37.7 | 44.4 | 40.3 | 42.1 | 48.2 | 54.9 | 44.4 | 42.1 | 54.6 | 58.0 | 45.1 | 46.4 | 47.6 | 36.4 | 40.4 | 45.5|
    | pytorch version | 35.7 | 42.3 | 39.4 | 40.7 | 44.5 | 53.3 | 42.8 | 40.1 | 52.5 | 53.9 | 42.8 | 43.1 | 44.1 | 33.4 | 36.3 | - |

## License
MIT
