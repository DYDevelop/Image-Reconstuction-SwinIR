# Training and testing codes for SwinIR
----------
- **TestSet(Left) and Transfer Leraning on Iter 95000 (Right)**
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
https://github.com/DYDevelop/SwinIR.git
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
@inproceedings{liang2021swinir,
title={SwinIR: Image Restoration Using Swin Transformer},
author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
booktitle={IEEE International Conference on Computer Vision Workshops},
pages={1833--1844},
year={2021}
}
```
Credits
----------
Our Swin Image Reconstruction implementation is heavily based on [Kai Zhang](https://github.com/cszn)'s [KAIR](https://github.com/cszn/KAIR).<br>
