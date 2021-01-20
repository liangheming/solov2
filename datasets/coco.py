import os
import torch
from typing import List
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
from utils.augmentations import *


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


coco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
colors = [(67, 68, 113), (130, 45, 169), (2, 202, 130), (127, 111, 90), (92, 136, 113),
          (33, 250, 7), (238, 92, 104), (0, 151, 197), (134, 9, 145), (253, 181, 88),
          (246, 11, 137), (55, 72, 220), (136, 8, 253), (56, 73, 180), (85, 241, 53),
          (153, 207, 15), (187, 183, 180), (149, 32, 71), (92, 113, 184), (131, 7, 201),
          (56, 20, 219), (243, 201, 77), (13, 74, 96), (79, 14, 44), (195, 150, 66),
          (2, 249, 42), (195, 135, 43), (105, 70, 66), (120, 107, 116), (122, 241, 22),
          (17, 19, 179), (162, 185, 124), (31, 65, 117), (88, 200, 80), (232, 49, 154),
          (72, 1, 46), (59, 144, 187), (200, 193, 118), (123, 165, 219), (194, 84, 34),
          (91, 184, 108), (252, 64, 153), (251, 121, 27), (105, 93, 210), (89, 85, 81),
          (58, 12, 154), (81, 3, 50), (200, 40, 236), (155, 147, 180), (73, 29, 176),
          (193, 19, 175), (157, 225, 121), (128, 195, 235), (146, 251, 108), (13, 146, 186),
          (231, 118, 145), (253, 15, 105), (187, 149, 62), (121, 247, 158), (34, 8, 142),
          (83, 61, 48), (119, 218, 69), (197, 94, 130), (222, 176, 142), (21, 20, 77),
          (6, 42, 17), (136, 33, 156), (39, 252, 211), (52, 50, 40), (183, 115, 34),
          (107, 80, 164), (195, 215, 74), (7, 154, 135), (136, 35, 24), (131, 241, 125),
          (208, 99, 208), (5, 4, 129), (137, 156, 175), (29, 141, 67), (44, 20, 99)]

rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]
cv.setNumThreads(0)

default_aug_cfg = {
    'hsv_h': 0.014,
    'hsv_s': 0.68,
    'hsv_v': 0.36,
    'degree': (-10, 10),
    'translate': 0,
    'shear': 0.0,
    'beta': (8, 8),
}


class COCODataSets(Dataset):

    def __init__(self,
                 img_root,
                 annotation_path,
                 min_threshes=None,
                 max_thresh=768,
                 augments=True,
                 use_crowd=False,
                 debug=False,
                 remove_blank=True,
                 aug_cfg=None,
                 square_padding=False):
        """
        :param img_root:
        :param annotation_path:
        :param min_threshes:
        :param max_thresh:
        :param augments:
        :param use_crowd:
        :param debug:
        :param remove_blank:
        :param aug_cfg:
        """
        super(COCODataSets, self).__init__()
        self.coco = COCO(annotation_path)
        if min_threshes is None:
            min_threshes = [416, 448, 480, 512, 544, 576, 608, 640]
        self.min_threshes = min_threshes
        self.max_thresh = max_thresh
        self.img_root = img_root
        self.use_crowd = use_crowd
        self.remove_blank = remove_blank
        self.augments = augments
        self.square_padding = square_padding
        if aug_cfg is None:
            aug_cfg = default_aug_cfg
        self.aug_cfg = aug_cfg
        self.debug = debug
        self.empty_images_len = 0
        data_len = len(self.coco.imgs.keys())
        data_info_list = self.__load_data()
        self.data_info_list = data_info_list
        if len(data_info_list) != data_len:
            print("all data len:{:d} | valid data len:{:d}".format(data_len, len(data_info_list)))
        if self.debug:
            assert debug <= len(data_info_list), "not enough data to debug"
            print("debug")
            self.data_info_list = data_info_list[:debug]
        self.transform = None
        self.set_transform()

    def __load_data(self):
        data_info_list = list()
        for img_id in self.coco.imgs.keys():
            file_name = self.coco.imgs[img_id]['file_name']
            width, height = self.coco.imgs[img_id]['width'], self.coco.imgs[img_id]['height']
            file_path = os.path.join(self.img_root, file_name)
            if not os.path.exists(file_path):
                print("img {:s} is not exist".format(file_path))
                continue
            assert width > 1 and height > 1, "invalid width or heights"
            anns = self.coco.imgToAnns[img_id]
            label_list = list()
            for ann in anns:
                category_id, box, iscrowd = ann['category_id'], ann['bbox'], ann['iscrowd']
                label_id = coco_ids.index(category_id)
                assert label_id >= 0, 'error label_id'
                if not self.use_crowd and iscrowd == 1:
                    continue
                x1, y1 = box[:2]
                x2, y2 = x1 + box[2], y1 + box[3]
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    print("not a valid box ", box)
                    continue
                if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
                    print("warning box ", box)
                label_list.append((label_id, (x1, y1, x2, y2), ann))
            valid_box_len = len(label_list)
            if valid_box_len == 0:
                box_seg_info = BoxSegInfo(img_path=file_path,
                                          boxes=np.zeros((0, 4)),
                                          labels=np.zeros((0,)),
                                          shape=(width, height),
                                          mask=np.zeros((0, height, width)))
            else:
                labels = np.array([i[0] for i in label_list])
                boxes = np.array([i[1] for i in label_list])
                masks_ann = [i[2] for i in label_list]
                box_seg_info = BoxSegInfo(img_path=file_path,
                                          boxes=boxes,
                                          labels=labels,
                                          shape=(width, height),
                                          coco_mask_ann=masks_ann)
                # print(boxes.shape, masks.shape, labels.shape)
            if self.remove_blank and valid_box_len == 0:
                self.empty_images_len += 1
                continue
            data_info_list.append(box_seg_info)
        return data_info_list

    def __getitem__(self, index):
        data_info = self.data_info_list[index].clone().load_img().load_mask(self.coco.annToMask)
        # data_info = RandScaleToMax(max_threshes=[640], pad_to_square=True)(data_info)
        data_info = self.transform(data_info)
        # assert data_info.img.dtype == np.uint8
        # import uuid
        # ret_img = data_info.draw_mask(colors, coco_names)
        # file_name = str(uuid.uuid4()).replace("-", "")
        # cv.imwrite("{:s}.jpg".format(file_name), ret_img)
        return data_info

    def set_transform(self):
        color_gitter = OneOf(
            transforms=[
                Identity(),
                RandHSV(hgain=self.aug_cfg['hsv_h'],
                        vgain=self.aug_cfg['hsv_v'],
                        sgain=self.aug_cfg['hsv_s']),
                RandBlur().reset(p=0.5),
                RandNoise().reset(p=0.5)
            ]
        )
        basic_transform = Compose(
            transforms=[
                color_gitter,
                RandCrop(min_thresh=0.6, max_thresh=1.0).reset(p=0.2),
                RandScaleMinMax(min_threshes=self.min_threshes, max_thresh=self.max_thresh),
                RandPerspective(degree=self.aug_cfg['degree'], scale=(1.0, 1.0))
            ]
        )

        mosaic = MosaicWrapper(candidate_box_info=self.data_info_list,
                               sizes=[self.max_thresh],
                               color_gitter=color_gitter,
                               annToMask=self.coco.annToMask)

        augment_transform = Compose(
            transforms=[
                OneOf(transforms=[
                    (1.0, basic_transform),
                    (0.0, mosaic)
                ]),
                LRFlip().reset(p=0.5)
            ]
        )
        std_transform = RandScaleMinMax(min_threshes=[640], max_thresh=self.max_thresh)

        if self.augments:
            self.transform = augment_transform
        else:
            self.transform = std_transform

    def __len__(self):
        return len(self.data_info_list)

    def collect_fn(self, batch: List[BoxSegInfo]):
        batch_img = list()
        batch_target = list()
        batch_length = list()
        batch_mask = list()
        valid_size = list()
        max_h = make_divisible(max([item.img.shape[0] for item in batch]), 64)
        max_w = make_divisible(max([item.img.shape[1] for item in batch]), 64)
        # if self.square_padding:
        #     max_h = max(max_w, max_h)
        #     max_w = max_h
        for item in batch:
            padding_val = item.padding_val
            img = np.ones((max_h, max_w, 3)) * np.array(padding_val)
            ori_img = item.img
            ori_h, ori_w = ori_img.shape[:2]
            img[0:ori_h, 0:ori_w, :] = ori_img
            img = (img[:, :, ::-1] / 255.0 - np.array(rgb_mean)) / np.array(rgb_std)
            batch_img.append(img)
            target = np.concatenate([item.labels[:, None], item.boxes], axis=-1)
            target_len = len(target)
            batch_target.append(target)
            batch_length.append(target_len)
            valid_size.append((ori_w, ori_h))
            # valid_size.append((max_w, max_h))
            if target_len == 0:
                continue
            mask = np.zeros(shape=(target_len, max_h, max_w))
            ori_mask = item.mask
            mask[:, 0:ori_h, 0:ori_w] = ori_mask
            batch_mask.append(mask)
        batch_img = torch.from_numpy(np.stack(batch_img, axis=0)).permute(0, 3, 1, 2).contiguous().float()
        batch_mask = torch.from_numpy(np.concatenate(batch_mask, axis=0)).float()
        batch_target = torch.from_numpy(np.concatenate(batch_target, axis=0)).float()
        return batch_img, valid_size, batch_target, batch_mask, batch_length


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    coco_data_sets = COCODataSets(img_root="/home/huffman/data/coco/val2017",
                                  annotation_path="/home/huffman/data/coco/annotations/instances_val2017.json",
                                  debug=50, augments=True, min_threshes=[640, ], max_thresh=640)
    loader = DataLoader(dataset=coco_data_sets, batch_size=2, shuffle=True, collate_fn=coco_data_sets.collect_fn)

    for i, s, t, m, l in loader:
        print(i.shape, m.shape, t.shape)
        print(s)
