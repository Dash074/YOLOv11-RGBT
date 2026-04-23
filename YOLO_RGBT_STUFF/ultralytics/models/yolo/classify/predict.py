# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from YOLO_RGBT_STUFF.ultralytics.utils import DEFAULT_CFG
import cv2
import numpy as np
import torch
from PIL import Image

from YOLO_RGBT_STUFF.ultralytics.engine.predictor import BasePredictor
from YOLO_RGBT_STUFF.ultralytics.engine.results import Results
from YOLO_RGBT_STUFF.ultralytics.utils import ops
import torchvision.transforms as T


class ClassificationPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a classification model.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes ClassificationPredictor setting the task to 'classify'."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"
        self._legacy_transform_name = "ultralytics.yolo.data.augment.ToTensor"

        self.model_channels = self.args.channels

    def preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            is_legacy_transform = any(
                self._legacy_transform_name in str(transform) for transform in self.transforms.transforms
            )

            if is_legacy_transform:  # to handle legacy transforms
                img = torch.stack([self.transforms(im) for im in img], dim=0)
            else:
                # 直接处理numpy数组，避免PIL转换问题
                processed_imgs = []
                for im in img:
                    # 确保图像是numpy数组
                    if not isinstance(im, np.ndarray):
                        im = np.array(im)

                    # 获取当前图像的通道数
                    current_channels = im.shape[2] if len(im.shape) == 3 else 1

                    # 严格根据模型通道数处理图像
                    if self.model_channels == 1:
                        # 1通道模型：只处理1通道图像
                        if len(im.shape) == 2:
                            im_processed = im  # 保持灰度图
                        elif current_channels == 1:
                            im_processed = im[:, :, 0]  # 单通道取第一通道
                        else:
                            # 多通道转灰度
                            im_processed = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                    elif self.model_channels == 3:
                        # 3通道模型：只处理3通道图像
                        if len(im.shape) == 2:
                            # 灰度图转3通道
                            im_processed = np.stack([im] * 3, axis=2)
                        elif current_channels == 1:
                            # 单通道转3通道
                            im_processed = np.repeat(im, 3, axis=2)
                        elif current_channels == 3:
                            # 3通道保持原样
                            im_processed = im
                        elif current_channels == 4:
                            # 4通道取前3通道
                            im_processed = im[:, :, :3]
                        elif current_channels == 6:
                            # 6通道取前3通道
                            im_processed = im[:, :, :3]
                        else:
                            # 其他通道数取前3通道
                            im_processed = im[:, :, :3]

                    elif self.model_channels == 4:
                        # 4通道模型：只处理4通道图像
                        if len(im.shape) == 2:
                            # 灰度图无法转换为4通道，报错
                            raise ValueError("1-channel image cannot be converted to 4 channels for 4-channel model")
                        elif current_channels == 1:
                            # 单通道无法转换为4通道，报错
                            raise ValueError("1-channel image cannot be converted to 4 channels for 4-channel model")
                        elif current_channels == 3:
                            # 3通道无法转换为4通道，报错
                            raise ValueError("3-channel image cannot be converted to 4 channels for 4-channel model")
                        elif current_channels == 4:
                            # 4通道保持原样
                            im_processed = im
                        elif current_channels == 6:
                            # 6通道取前4通道
                            im_processed = im[:, :, :4]
                        else:
                            # 其他通道数取前4通道
                            im_processed = im[:, :, :min(current_channels, 4)]
                            if current_channels < 4:
                                raise ValueError(
                                    f"{current_channels}-channel image cannot be converted to 4 channels for 4-channel model")

                    elif self.model_channels == 6:
                        # 6通道模型：只处理6通道图像
                        if len(im.shape) == 2:
                            # 灰度图无法转换为6通道，报错
                            raise ValueError("1-channel image cannot be converted to 6 channels for 6-channel model")
                        elif current_channels == 1:
                            # 单通道无法转换为6通道，报错
                            raise ValueError("1-channel image cannot be converted to 6 channels for 6-channel model")
                        elif current_channels == 3:
                            # 3通道无法转换为6通道，报错
                            raise ValueError("3-channel image cannot be converted to 6 channels for 6-channel model")
                        elif current_channels == 4:
                            # 4通道无法转换为6通道，报错
                            raise ValueError("4-channel image cannot be converted to 6 channels for 6-channel model")
                        elif current_channels == 6:
                            # 6通道保持原样
                            im_processed = im
                        else:
                            # 其他通道数取前6通道
                            im_processed = im[:, :, :min(current_channels, 6)]
                            if current_channels < 6:
                                raise ValueError(
                                    f"{current_channels}-channel image cannot be converted to 6 channels for 6-channel model")

                    else:
                        # 其他通道数模型：严格匹配
                        if current_channels != self.model_channels:
                            raise ValueError(
                                f"{current_channels}-channel image does not match {self.model_channels}-channel model")
                        im_processed = im

                    # 应用几何变换
                    im_processed = self._apply_geometric_transforms(im_processed)

                    # 转换为tensor并归一化到[0,1]
                    tensor_img = torch.from_numpy(im_processed.transpose(2, 0, 1)).float() / 255.0

                    # 应用归一化
                    tensor_img = self._apply_normalization(tensor_img)

                    processed_imgs.append(tensor_img)

                img = torch.stack(processed_imgs, dim=0)

        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()

    def _apply_geometric_transforms(self, img):
        """应用几何变换（resize和centercrop）"""
        # 获取transform参数
        import torchvision.transforms as T

        # 查找resize参数
        resize_size = 224  # 默认值
        for transform in self.transforms.transforms:
            if isinstance(transform, T.Resize):
                if isinstance(transform.size, int):
                    resize_size = (transform.size, transform.size)
                else:
                    resize_size = transform.size
                break

        # 应用resize
        h, w = img.shape[:2]
        if isinstance(resize_size, int):
            resize_size = (resize_size, resize_size)

        if (h, w) != resize_size:
            # 根据模型通道数采用不同的resize策略
            if self.model_channels == 1:
                # 1通道：灰度resize
                img = cv2.resize(img, resize_size, interpolation=cv2.INTER_LINEAR)
            elif self.model_channels == 3:
                # 3通道：彩色resize
                img = cv2.resize(img, resize_size, interpolation=cv2.INTER_LINEAR)
            elif self.model_channels == 4:
                # 4通道：前3通道彩色resize，第4通道灰度resize
                rgb_resized = cv2.resize(img[:, :, :3], resize_size, interpolation=cv2.INTER_LINEAR)
                fourth_resized = cv2.resize(img[:, :, 3], resize_size, interpolation=cv2.INTER_LINEAR)
                img = np.concatenate([rgb_resized, fourth_resized[:, :, np.newaxis]], axis=2)
            elif self.model_channels == 6:
                # 6通道：分别处理两个RGB组
                rgb1_resized = cv2.resize(img[:, :, :3], resize_size, interpolation=cv2.INTER_LINEAR)
                rgb2_resized = cv2.resize(img[:, :, 3:6], resize_size, interpolation=cv2.INTER_LINEAR)
                img = np.concatenate([rgb1_resized, rgb2_resized], axis=2)
            else:
                # 其他通道数：标准resize
                img = cv2.resize(img, resize_size, interpolation=cv2.INTER_LINEAR)

        # 查找centercrop参数
        for transform in self.transforms.transforms:
            if isinstance(transform, T.CenterCrop):
                crop_size = transform.size
                if isinstance(crop_size, int):
                    crop_size = (crop_size, crop_size)

                h, w = img.shape[:2]
                if h >= crop_size[0] and w >= crop_size[1]:
                    top = (h - crop_size[0]) // 2
                    left = (w - crop_size[1]) // 2
                    img = img[top:top + crop_size[0], left:left + crop_size[1]]
                break

        return img

    def _apply_normalization(self, tensor):
        """应用归一化"""
        # 查找归一化参数
        for transform in self.transforms.transforms:
            if isinstance(transform, T.Normalize):
                mean = transform.mean
                std = transform.std

                # 根据模型通道数严格应用归一化
                tensor_channels = tensor.shape[0]

                if tensor_channels == len(mean):
                    # 通道数完全匹配，直接应用
                    for t, m, s in zip(tensor, mean, std):
                        t.sub_(m).div_(s)
                elif len(mean) == 3 and tensor_channels != 3:
                    # 3通道归一化参数应用到多通道模型
                    if tensor_channels == 1:
                        # 1通道模型：取平均值
                        avg_mean = sum(mean) / 3.0
                        avg_std = sum(std) / 3.0
                        tensor.sub_(avg_mean).div_(avg_std)
                    elif tensor_channels == 4:
                        # 4通道模型：前3通道用RGB参数，第4通道用默认
                        for i in range(3):
                            tensor[i].sub_(mean[i]).div_(std[i])
                        tensor[3].sub_(0.0).div_(1.0)
                    elif tensor_channels == 6:
                        # 6通道模型：两组RGB分别应用相同的归一化
                        for i in range(3):
                            tensor[i].sub_(mean[i]).div_(std[i])  # 第一组RGB
                            tensor[i + 3].sub_(mean[i]).div_(std[i])  # 第二组RGB
                else:
                    # 不匹配的情况：使用默认归一化
                    for i in range(tensor_channels):
                        if i < len(mean):
                            tensor[i].sub_(mean[i]).div_(std[i])
                        else:
                            tensor[i].sub_(0.0).div_(1.0)
                break

        return tensor

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        preds = preds[0] if isinstance(preds, (list, tuple)) else preds

        # 处理多通道图像的可视化问题
        processed_orig_imgs = []
        for orig_img in orig_imgs:
            # 如果原始图像是多通道（>3），转换为3通道用于可视化
            if len(orig_img.shape) == 3 and orig_img.shape[2] > 3:
                if orig_img.shape[2] == 4:
                    # 4通道：取前3通道用于可视化
                    vis_img = orig_img[:, :, :3]
                elif orig_img.shape[2] == 6:
                    # 6通道：取第一组RGB用于可视化
                    vis_img = orig_img[:, :, :3]
                else:
                    # 其他多通道：取前3通道
                    vis_img = orig_img[:, :, :3]
                processed_orig_imgs.append(vis_img)
            else:
                processed_orig_imgs.append(orig_img)

        return [
            Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            for pred, orig_img, img_path in zip(preds, processed_orig_imgs, self.batch[0])
        ]