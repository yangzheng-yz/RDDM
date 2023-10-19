import os
import random
from pathlib import Path

import Augmentor
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size=(512, 640),
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_flip=False,
        convert_image_to=None,
        condition=0,
        equalizeHist=False,
        crop_patch=True,
        sample=False,
        artifact_property={'random': True, 'fixed_exposure_time': 100}
    ):
        super().__init__()
        self.equalizeHist = equalizeHist
        self.exts = exts
        self.augment_flip = augment_flip
        self.condition = condition
        self.crop_patch = crop_patch
        self.sample = sample
        if condition == 1:
            # condition (this should be for denoising)
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
        elif condition == 11: # TODO: align this No. in all files
            self.gt = self.load_png(folder[0])
        elif condition == 0:
            # generation
            self.paths = self.load_flist(folder)
        elif condition == 2:
            self.gt = self.load_flist(folder[0])
            self.input = self.load_flist(folder[1])
            self.input_condition = self.load_flist(folder[2])

        self.image_size = image_size
        self.convert_image_to = convert_image_to
        self.artifact_property = artifact_property
        if self.artifact_property.get('artifact_source_path', None) is not None:
            self.artifacts_source = self.read_src_artifacts(self.artifact_property['artifact_source_path'],
                                                            image_shape=(512, 640), dtype=np.uint8, num_images=268)
        else:
            self.artifacts_source = None
        self.ept = artifact_property.get('fixed_exposure_time', 100)

    def __len__(self):
        if self.condition:
            if self.condition == 11:
                return len(self.gt)
            else:
                return len(self.input)
        else:
            return len(self.paths)

    def __getitem__(self, index):
        if self.condition == 1:
            # condition
            img0 = Image.open(self.gt[index])
            img1 = Image.open(self.input[index])
            w, h = img0.size
            img0 = convert_image_to_fn(
                self.convert_image_to, img0) if self.convert_image_to else img0
            img1 = convert_image_to_fn(
                self.convert_image_to, img1) if self.convert_image_to else img1

            img0, img1 = self.pad_img([img0, img1], self.image_size)

            if self.crop_patch and not self.sample:
                img0, img1 = self.get_patch([img0, img1], self.image_size)

            img1 = self.cv2equalizeHist(img1) if self.equalizeHist else img1

            images = [[img0, img1]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            if not self.crop_patch:
                p.resize(1, self.image_size, self.image_size)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            img0 = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)

            return [self.to_tensor(img0), self.to_tensor(img1)]
        elif self.condition == 11:
            # condition
            # print(f"Debug self.gt[index] is {self.gt[index]}")
            # print(f"Debug self.gt[index] is {self.gt[index]}"
            # with Image.open(self.gt[index]) as img:
            #     img0 = img.convert("L")
            img0 = Image.open(self.gt[index])
            # img1 = Image.open(self.gt[index])
            # w, h = img0.size
            # print(f"Debug in base.py Dataset Class img0(gt) width: {w}, height: {h}")
            if "0002_20ms" in self.gt[index] or "0003_50ms" in self.gt[index]: 
                img1 = img0.copy()
                self.ept = int(self.gt[index].split("_")[-1].split('ms')[0])
            else:
                img1, ept = self.generate_artifacted_image(img0.copy(), artifacts_source=self.artifacts_source, \
                    r=self.artifact_property['random'], exposure_time=self.artifact_property.get('fixed_exposure_time', None))
                self.ept = ept
            # print(f"Debug in base.py Dataset Class img1(input) width: {img1.size[0]}, height: {img1.size[1]}")
            
            img0 = convert_image_to_fn(
                self.convert_image_to, img0) if self.convert_image_to else img0
            img1 = convert_image_to_fn(
                self.convert_image_to, img1) if self.convert_image_to else img1
            # print(f"Debug in base.py Dataset Class img0(input) width after convert_image_to_fn: {img0.size[0]}, height: {img0.size[1]}")
            # print(f"Debug in base.py Dataset Class img1(input) width after convert_image_to_fn: {img1.size[0]}, height: {img1.size[1]}")

            img0, img1 = self.pad_img([img0, img1], self.image_size) if self.convert_image_to != "L" else self.pad_img_L([img0,img1], self.image_size)
            # print(f"Debug in base.py Dataset Class img0(input) width after pad_img: {img0.shape}, height: {img0.shape}")
            # print(f"Debug in base.py Dataset Class img1(input) width after pad_img: {img1.shape}, height: {img1.shape}")

            if self.crop_patch and not self.sample:
                img0, img1 = self.get_patch([img0, img1], self.image_size)
            # print(f"Debug in base.py Dataset Class img0(input) width after get_patch: {img0.shape}, height: {img0.shape}")
            # print(f"Debug in base.py Dataset Class img1(input) width after get_patch: {img1.shape}, height: {img1.shape}")

            img1 = self.cv2equalizeHist(img1) if self.equalizeHist else img1

            images = [[img0, img1]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            if not self.crop_patch:
                p.resize(1, self.image_size, self.image_size)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            # print(f"Debug in base.py Dataset Class img0(input) width after augmented_images: {augmented_images[0][0].shape}, height: {augmented_images[0][0].shape}")
            # print(f"Debug in base.py Dataset Class img1(input) width after augmented_images: {augmented_images[0][0].shape}, height: {augmented_images[0][0].shape}")
            img0 = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB) 
            img1 = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB) 
            if self.convert_image_to == "L":
                img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            # print(f"Debug in base.py Dataset Class img0(input) width in the final _get_item_: {img0.shape}, height: {img0.shape}")
            # print(f"Debug in base.py Dataset Class img1(input) width in the final _get_item_: {img1.shape}, height: {img1.shape}")
            return [self.to_tensor(img0), self.to_tensor(img1)]        
        elif self.condition == 0:
            # generation
            path = self.paths[index]
            img = Image.open(path)
            img = convert_image_to_fn(
                self.convert_image_to, img) if self.convert_image_to else img

            img = self.pad_img([img], self.image_size)[0]

            if self.crop_patch and not self.sample:
                img = self.get_patch([img], self.image_size)[0]

            img = self.cv2equalizeHist(img) if self.equalizeHist else img

            images = [[img]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            if not self.crop_patch:
                p.resize(1, self.image_size, self.image_size)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            img = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)

            return self.to_tensor(img)
        elif self.condition == 2:
            # condition
            img0 = Image.open(self.gt[index])
            img1 = Image.open(self.input[index])
            img2 = Image.open(self.input_condition[index])
            img0 = convert_image_to_fn(
                self.convert_image_to, img0) if self.convert_image_to else img0
            img1 = convert_image_to_fn(
                self.convert_image_to, img1) if self.convert_image_to else img1
            img2 = convert_image_to_fn(
                self.convert_image_to, img2) if self.convert_image_to else img2

            img0, img1, img2 = self.pad_img(
                [img0, img1, img2], self.image_size)

            if self.crop_patch and not self.sample:
                img0, img1, img2 = self.get_patch(
                    [img0, img1, img2], self.image_size)

            img1 = self.cv2equalizeHist(img1) if self.equalizeHist else img1

            images = [[img0, img1, img2]]
            p = Augmentor.DataPipeline(images)
            if self.augment_flip:
                p.flip_left_right(1)
            if not self.crop_patch:
                p.resize(1, self.image_size, self.image_size)
            g = p.generator(batch_size=1)
            augmented_images = next(g)
            img0 = cv2.cvtColor(augmented_images[0][0], cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(augmented_images[0][1], cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(augmented_images[0][2], cv2.COLOR_BGR2RGB)

            return [self.to_tensor(img0), self.to_tensor(img1), self.to_tensor(img2)]

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                return [p for ext in self.exts for p in Path(f'{flist}').glob(f'**/*.{ext}')]

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def load_png(self, dir_path):
        assert isinstance(dir_path, str), "Should be a absolute path to the data folder."
        return [os.path.join(dir_path, i) for i in os.listdir(dir_path)]

    def cv2equalizeHist(self, img):
        (b, g, r) = cv2.split(img)
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
        img = cv2.merge((b, g, r))
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t

    def load_name(self, index, sub_dir=False):
        if self.condition:
            # condition
            try:
                name = self.input[index]
            except:
                name = self.gt[index]
            if sub_dir == 0:
                # print(f"Debug name: {name}")
                file_name = os.path.basename(name)
                if self.condition == 11:
                    file_name = f'{file_name.split(".")[0]}_ept{self.ept}.png'
                return file_name
            elif sub_dir == 1:
                path = os.path.dirname(name)
                sub_dir = (path.split("/"))[-1]
                return sub_dir+"_"+os.path.basename(name)

    def get_patch(self, image_list, patch_size):
        i = 0
        h, w = image_list[0].shape[:2]
        rr = random.randint(0, h-patch_size)
        cc = random.randint(0, w-patch_size)
        for img in image_list:
            image_list[i] = img[rr:rr+patch_size, cc:cc+patch_size, :]
            i += 1
        return image_list

    def pad_img(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size-h
                h = patch_size
            if w < patch_size:
                right = patch_size-w
                w = patch_size
            bottom = bottom + (h // block_size) * block_size + \
                (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            i += 1
        return img_list

    def pad_img_L(self, img_list, patch_size, block_size=8):
        i = 0
        for img in img_list:
            # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = np.asarray(img)
            h, w = img.shape[:2]
            bottom = 0
            right = 0
            if h < patch_size:
                bottom = patch_size-h
                h = patch_size
            if w < patch_size:
                right = patch_size-w
                w = patch_size
            bottom = bottom + (h // block_size) * block_size + \
                (block_size if h % block_size != 0 else 0) - h
            right = right + (w // block_size) * block_size + \
                (block_size if w % block_size != 0 else 0) - w
            img_list[i] = cv2.copyMakeBorder(
                img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0])
            i += 1
        return img_list

    def get_pad_size(self, index, block_size=8):
        img = Image.open(self.input[index])
        patch_size = self.image_size
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        bottom = 0
        right = 0
        if h < patch_size:
            bottom = patch_size-h
            h = patch_size
        if w < patch_size:
            right = patch_size-w
            w = patch_size
        bottom = bottom + (h // block_size) * block_size + \
            (block_size if h % block_size != 0 else 0) - h
        right = right + (w // block_size) * block_size + \
            (block_size if w % block_size != 0 else 0) - w
        return [bottom, right]

    def generate_artifacted_image(self, src, artifacts_source=None, r=False, exposure_time=100):
        assert artifacts_source is not None, "You must generate the artifacts source list first. "
        
        if r:
            threshold = random.random()  # 生成 0 到 1 之间的随机浮点数
            if threshold <= 1:
                # 50% 的概率在 0-249 之间生成随机数
                list_index = random.randint(0, 249)
            else:
                # 50% 的概率在 250-267 之间生成随机数
                list_index = random.randint(250, 267)
            # list_index = random.randint(0, len(artifacts_source)-1)
            if list_index >= 0 and list_index < 250:
                exposure_time = 2 + 2 * list_index
            elif list_index >= 250 and list_index < 260:
                exposure_time = 1050 - 50 * (260 - list_index)
            else:
                exposure_time = 5500 - 500 * (268 - list_index)
        else:
            if exposure_time >= 2 and exposure_time <= 500:
                list_index = int(exposure_time / 2) - 1
            elif exposure_time > 500 and exposure_time <= 1000:
                list_index = int( (exposure_time - 500) / 50 ) + 249
            else:
                list_index = int( (exposure_time - 1000) / 500 ) + 259
        # src = np.array(src)
        # overlay_image = src.copy().astype(np.float32) / 255.0
        # decrease_means = np.random.uniform(0.2, 0.7, size=overlay_image.shape)
        # noise = np.random.normal(loc=decrease_means, scale=0.05, size=overlay_image.shape)
        # mean_adjustment = 2.0 * (0.5 - overlay_image)
        # mask_adjustment = -np.log(np.abs(1.0 + 1e-6 - overlay_image))
        # mask = np.random.normal(loc=0.5 + np.abs(mean_adjustment), scale=0.05, size=overlay_image.shape)
        # outliers = np.where(artifacts_source[list_index] < 50)
        # mask *= mask_adjustment
        # mask[outliers] = 0.0
        # noise *= mask
        # overlay_image -= noise
        # overlay_image[overlay_image < 0] += 1.0
        # overlay_image = np.clip(overlay_image, 0, 1)
        # overlay_image = (overlay_image * 255)
        
        # # cv2.imwrite(f"./Intermedium_results/{exposure_time}ms.png", dst)
        # dst = Image.fromarray(overlay_image)
        # if dst.mode != "L":
        #     dst = dst.convert("L")
        # dst.save("./100ms.png")
        # return dst, exposure_time

        src = np.array(src)
        normal_values = np.random.normal(loc=0, scale=1, size=src.shape)
        mask = np.zeros_like(src)
        const = np.random.normal(loc=0.5, scale=0, size=src.shape)
        outliers = np.where(artifacts_source[list_index] >= 50)
        mask[outliers] = 1.0
        normal_values *= mask
        const *= mask
        if np.max(src) > 1:
            dst = (src + 0.6 * (normal_values - const) * 255).astype(np.uint8)
            dst = np.clip(dst, 0, 255) 
        else:
            dst = (src + 0.6 * (normal_values - const)).astype(np.float32)
            dst = np.clip(dst, 0, 1) 
        
        # cv2.imwrite(f"./Intermedium_results/{exposure_time}ms.png", dst)
        dst = Image.fromarray(dst)
        # if dst.mode != "L":
        #     dst = dst.convert("L")
        dst.save("./100ms.png")
        return dst, exposure_time
    
    def read_src_artifacts(self, path_to_raw, image_shape=(512, 640), dtype=np.uint8, num_images=268):
        return self.read_raw_images(path_to_raw, image_shape, dtype=np.uint8, num_images=268)

    def read_raw_images(self, path_to_raw, shape, dtype=np.uint8, num_images=268, big_endian=False):
        image_bytes = np.prod(shape) * np.dtype(dtype).itemsize * num_images
        with open(path_to_raw, 'rb') as f:
            img_data = np.frombuffer(f.read(image_bytes), dtype=dtype)
        if big_endian:
            img_data = img_data.byteswap()
        images = img_data.reshape((num_images, *shape)).copy()

        return images

if __name__ == "__main__":
    dataset = Dataset(["/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/val/without_art"],
                       64,
                       exts=['png'],
                       augment_flip=False,
                       convert_image_to="L",
                       condition=11,
                       equalizeHist=False,
                       crop_patch=False,
                       sample=False,
                       artifact_property={'random': False, 'fixed_exposure_time': 100, 'artifact_source_path': "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/For_argumentation/et_2-500ms-250_550-1000ms-10_1500-5000ms-8_1800LP_contrast.raw"})
    
    img = dataset[-1]
    
    