# Monocular 3D Pose Esitimation Model Toolbox  

1. [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) (CVPR 2019) 
2. [PoseFormer](https://github.com/zczcwh/PoseFormer) (ICCV 2021)
3. [MHFormer](https://github.com/karfly/learnable-triangulation-pytorch) (CVPR 2022) 
4. [StridedTransformer](https://github.com/Vegetebird/StridedTransformer-Pose3D) (TMM 2022) 
5. [Anatomy3D](https://github.com/sunnychencool/Anatomy3D) (TCSVT 2021) 

## Dataset 
Please download the dataset from [Human3.6M website](http://vision.imar.ro/human3.6m/description.php) and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). Or you can download the processed data from [here](https://drive.google.com/drive/folders/112GPdRC9IEcwcJRyrLJeYw9_YV4wLdKC?usp=sharing).

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

## Download pretrained models
Please download pretrained models and put them in the './checkpoint/pretrained' directory.

* [PoseFormer](https://drive.google.com/file/d/1oX5H5QpVoFzyD-Qz9aaP3RDWDb1v1sIy/view?usp=sharing)
* [HMFormer](https://drive.google.com/drive/folders/1UWuaJ_nE19x2aM-Th221UpdhRPSCFwZa?usp=sharing)
* [StridedTransformer](https://drive.google.com/drive/folders/13p62U_f5Z5Ay2jyRlXDkeU7tccde8DxS?usp=sharing)
* [Anatomy3D](https://drive.google.com/file/d/17QIbAWfCP5fwiz9MhFw1pRBZ_2VlgMSU/view?usp=sharing)


## License
This project is licensed under the terms of the MIT license.
Third-party datasets are subject to their license.
