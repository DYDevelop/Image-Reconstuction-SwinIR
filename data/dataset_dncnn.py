import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import cv2


class DatasetDnCNN(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDnCNN, self).__init__()
        print('Dataset: Denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            target_size, height, width = 128, H, W

            # 타겟 사이즈로 resize
            if height > width:
                new_height = target_size
                new_width = int(width * (target_size / height))
            else:
                new_width = target_size
                new_height = int(height * (target_size / width))

            resized_img = cv2.resize(img_H, (new_width, new_height))

            # 패딩을 위한 빈 이미지 생성
            padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

            # 패딩 위치 계산
            y_offset = (target_size - new_height) // 2
            x_offset = (target_size - new_width) // 2

            # 패딩된 이미지에 resize된 이미지 삽입
            padded_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized_img

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            # rnd_h = random.randint(0, max(0, H - self.patch_size))
            # rnd_w = random.randint(0, max(0, W - self.patch_size))
            # patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H = padded_img

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            # --------------------------------
            # add noise
            # --------------------------------
            noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)
            img_L.add_(noise)

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """

            H, W, _ = img_H.shape

            target_size, height, width = 128, H, W

            # 타겟 사이즈로 resize
            if height > width:
                new_height = target_size
                new_width = int(width * (target_size / height))
            else:
                new_width = target_size
                new_height = int(height * (target_size / width))

            resized_img = cv2.resize(img_H, (new_width, new_height))

            # 패딩을 위한 빈 이미지 생성
            padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

            # 패딩 위치 계산
            y_offset = (target_size - new_height) // 2
            x_offset = (target_size - new_width) // 2

            # 패딩된 이미지에 resize된 이미지 삽입
            padded_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized_img

            img_H = padded_img

            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)

            # --------------------------------
            # add noise
            # --------------------------------
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)


# python main_test_swinir.py --task color_dn --noise 0 --model_path denoising/swinir_denoising_color_15/models/80000_G.pth --folder_gt testsets/platesc