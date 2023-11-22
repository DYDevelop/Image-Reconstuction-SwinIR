## Training and testing codes for SwinIR
----------
- TestSet(Left) and Transfer Leraning on Iter 95000 (Right)
<img width="100%" src="https://github.com/DYDevelop/SwinIR/assets/55197580/5d835d9a-e21f-4f6c-a9dc-d57986c44aa4"/>
<img width="100%" src="https://github.com/DYDevelop/SwinIR/assets/55197580/23fa410a-2ddf-4af1-9ab9-464a38e0ed78"/>
<img width="100%" src="https://github.com/DYDevelop/SwinIR/assets/55197580/3f13fc7f-cdb9-49f6-b207-7e905335cb98"/>
<img width="100%" src="https://github.com/DYDevelop/SwinIR/assets/55197580/457cbe0b-2934-4c8c-a2b1-bc5319b9b830"/>
<img width="100%" src="https://github.com/DYDevelop/SwinIR/assets/55197580/7fb7b526-5212-4e9b-8598-0575b90cdb9a"/>
<img width="100%" src="https://github.com/DYDevelop/SwinIR/assets/55197580/8e7846fa-a837-4f32-8429-fdea145643a0"/>
<img width="100%" src="https://github.com/DYDevelop/SwinIR/assets/55197580/742c705a-d0c4-489b-9565-c0968bcbfd75"/>

Clone repo
----------
```
git clone https://github.com/cszn/KAIR.git
```
```
pip install -r requirement.txt
```



Training
----------

You should modify the json file from [options](https://github.com/cszn/KAIR/tree/master/options) first, for example,
setting ["gpu_ids": [0,1,2,3]](https://github.com/cszn/KAIR/blob/ff80d265f64de67dfb3ffa9beff8949773c81a3d/options/train_msrresnet_psnr.json#L4) if 4 GPUs are used,
setting ["dataroot_H": "trainsets/trainH"](https://github.com/cszn/KAIR/blob/ff80d265f64de67dfb3ffa9beff8949773c81a3d/options/train_msrresnet_psnr.json#L24) if path of the high quality dataset is `trainsets/trainH`.

- Training with `DataParallel` - SwinIR


```python
python main_train_psnr.py --opt options/swinir/train_swinir_denoising_color.json
```


Inference
----------

- Inference on `DataParallel` - SwinIR


```python
python main_test_swinir.py --task color_dn --noise 0 --model_path denoising/swinir_denoising_color_15/models/100000_G.pth --folder_gt testsets/custom_dataset
```


References
----------
```BibTex
@inproceedings{zhu2023denoising, % DiffPIR
title={Denoising Diffusion Models for Plug-and-Play Image Restoration},
author={Yuanzhi Zhu and Kai Zhang and Jingyun Liang and Jiezhang Cao and Bihan Wen and Radu Timofte and Luc Van Gool},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition Workshops},
year={2023}
}
@article{liang2022vrt,
title={VRT: A Video Restoration Transformer},
author={Liang, Jingyun and Cao, Jiezhang and Fan, Yuchen and Zhang, Kai and Ranjan, Rakesh and Li, Yawei and Timofte, Radu and Van Gool, Luc},
journal={arXiv preprint arXiv:2022.00000},
year={2022}
}
@inproceedings{liang2021swinir,
title={SwinIR: Image Restoration Using Swin Transformer},
author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
booktitle={IEEE International Conference on Computer Vision Workshops},
pages={1833--1844},
year={2021}
}
@inproceedings{zhang2021designing,
title={Designing a Practical Degradation Model for Deep Blind Image Super-Resolution},
author={Zhang, Kai and Liang, Jingyun and Van Gool, Luc and Timofte, Radu},
booktitle={IEEE International Conference on Computer Vision},
pages={4791--4800},
year={2021}
}
@article{zhang2021plug, % DPIR & DRUNet & IRCNN
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021}
}
@inproceedings{zhang2020aim, % efficientSR_challenge
  title={AIM 2020 Challenge on Efficient Super-Resolution: Methods and Results},
  author={Kai Zhang and Martin Danelljan and Yawei Li and Radu Timofte and others},
  booktitle={European Conference on Computer Vision Workshops},
  year={2020}
}
@inproceedings{zhang2020deep, % USRNet
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3217--3226},
  year={2020}
}
@article{zhang2017beyond, % DnCNN
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017}
}
@inproceedings{zhang2017learning, % IRCNN
title={Learning deep CNN denoiser prior for image restoration},
author={Zhang, Kai and Zuo, Wangmeng and Gu, Shuhang and Zhang, Lei},
booktitle={IEEE conference on computer vision and pattern recognition},
pages={3929--3938},
year={2017}
}
@article{zhang2018ffdnet, % FFDNet, FDnCNN
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018}
}
@inproceedings{zhang2018learning, % SRMD
  title={Learning a single convolutional super-resolution network for multiple degradations},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3262--3271},
  year={2018}
}
@inproceedings{zhang2019deep, % DPSR
  title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1671--1681},
  year={2019}
}
@InProceedings{wang2018esrgan, % ESRGAN, MSRResNet
    author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
    title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
    booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
    month = {September},
    year = {2018}
}
@inproceedings{hui2019lightweight, % IMDN
  title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
  author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
  pages={2024--2032},
  year={2019}
}
@inproceedings{zhang2019aim, % IMDN
  title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
  author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  year={2019}
}
@inproceedings{yang2021gan,
    title={GAN Prior Embedded Network for Blind Face Restoration in the Wild},
    author={Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
    year={2021}
}
```
