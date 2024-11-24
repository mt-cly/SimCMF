import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from utils import random_click
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from conf.global_settings import modality_datapath_map

def get_dataloader(args):
    g =torch.Generator()

    data_path = modality_datapath_map[args.modality]
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
    ])

    transform_train_seg = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.out_size, args.out_size),
                          interpolation=transforms.InterpolationMode.NEAREST),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
    ])

    transform_test_seg = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.out_size, args.out_size),
                          interpolation=transforms.InterpolationMode.NEAREST),
    ])

    if args.modality == "zju-rgbp":
        '''zju_rgbp data'''
        zju_rgbp_train_dataset = ZJU_RGBP(args, data_path, transform=transform_train,
                                          transform_msk=transform_train_seg, mode='train')
        zju_rgbp_test_dataset = ZJU_RGBP(args, data_path, transform=transform_test,
                                         transform_msk=transform_test_seg, mode='val')
        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(zju_rgbp_train_dataset)
            test_sampler = DistributedSampler(zju_rgbp_test_dataset)
        nice_train_loader = DataLoader(zju_rgbp_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(zju_rgbp_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)
        '''end'''

    elif args.modality == "nir":
        '''nir data'''
        nir_train_dataset = NIR(args, data_path, transform = transform_train,
                                   transform_msk= transform_train_seg, mode = 'train')

        nir_test_dataset =  NIR(args, data_path, transform = transform_test,
                                   transform_msk= transform_test_seg, mode = 'test')

        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(nir_train_dataset)
            test_sampler = DistributedSampler(nir_test_dataset)
        nice_train_loader = DataLoader(nir_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(nir_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)

    elif args.modality == "rgbnir":
        '''nir data'''
        nir_train_dataset = RGBNIR(args, data_path, transform = transform_train,
                                   transform_msk= transform_train_seg, mode = 'train')

        nir_test_dataset =  RGBNIR(args, data_path, transform = transform_test,
                                   transform_msk= transform_test_seg, mode = 'test')
        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(nir_train_dataset)
            test_sampler = DistributedSampler(nir_test_dataset)
        nice_train_loader = DataLoader(nir_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(nir_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)


    elif args.modality == "rgbt":
        '''rgb and thermal glass data'''
        nir_train_dataset = RGBThermal(args, data_path, transform=transform_train,
                                       transform_msk=transform_train_seg, mode='train')
        nir_test_dataset = RGBThermal(args, data_path, transform=transform_test, transform_msk=transform_test_seg,
                                      mode='test')
        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(nir_train_dataset)
            test_sampler = DistributedSampler(nir_test_dataset)
        nice_train_loader = DataLoader(nir_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(nir_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)

    elif args.modality == "t":
        '''rgb and thermal glass data'''
        nir_train_dataset = Thermal(args, data_path, transform=transform_train,
                                       transform_msk=transform_train_seg, mode='train')
        nir_test_dataset = Thermal(args, data_path, transform=transform_test, transform_msk=transform_test_seg,
                                      mode='test')
        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(nir_train_dataset)
            test_sampler = DistributedSampler(nir_test_dataset)
        nice_train_loader = DataLoader(nir_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(nir_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)

    elif args.modality == "rgbt_500":
        '''rgb and thermal glass data'''
        rgbt_train_dataset = RGBThermal_500(args, data_path, transform=transform_train,
                                       transform_msk=transform_train_seg, mode='train')
        rgbt_test_dataset = RGBThermal_500(args, data_path, transform=transform_test, transform_msk=transform_test_seg,
                                      mode='test')
        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(rgbt_train_dataset)
            test_sampler = DistributedSampler(rgbt_test_dataset)
        nice_train_loader = DataLoader(rgbt_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(rgbt_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)

    elif args.modality == "t_500":
        '''rgb and thermal glass data'''
        t_train_dataset = Thermal_500(args, data_path, transform=transform_train,
                                       transform_msk=transform_train_seg, mode='train')
        t_test_dataset = Thermal_500(args, data_path, transform=transform_test, transform_msk=transform_test_seg,
                                      mode='test')
        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(t_train_dataset)
            test_sampler = DistributedSampler(t_test_dataset)
        nice_train_loader = DataLoader(t_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(t_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)

    elif args.modality == "pgsnet_rgbp":
        pgsnet_train_dataset = PGSNet_RGBP(args, data_path, transform=transform_train,
                                      transform_msk=transform_train_seg, mode='train')
        pgsnet_test_dataset = PGSNet_RGBP(args, data_path, transform=transform_test, transform_msk=transform_test_seg,
                                     mode='test')
        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(pgsnet_train_dataset)
            test_sampler = DistributedSampler(pgsnet_test_dataset)
        nice_train_loader = DataLoader(pgsnet_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(pgsnet_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)

    elif args.modality == "pgsnet_p":
        pgsnet_train_dataset = PGSNet_P(args, data_path, transform=transform_train,
                                           transform_msk=transform_train_seg, mode='train')
        pgsnet_test_dataset = PGSNet_P(args, data_path, transform=transform_test,
                                          transform_msk=transform_test_seg,
                                          mode='test')
        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(pgsnet_train_dataset)
            test_sampler = DistributedSampler(pgsnet_test_dataset)
        nice_train_loader = DataLoader(pgsnet_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(pgsnet_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)

    elif args.modality == 'rgbd':
        nyudv2_train_dataset = NYUDv2_RGBD(data_path, transform=transform_train, transform_msk=transform_train_seg,
                                      mode='train')
        nyudv2_test_dataset = NYUDv2_RGBD(data_path, transform=transform_test, transform_msk=transform_test_seg,
                                     mode='test')
        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(nyudv2_train_dataset)
            test_sampler = DistributedSampler(nyudv2_test_dataset)
        nice_train_loader = DataLoader(nyudv2_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(nyudv2_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)

    elif args.modality == 'd':
        nyudv2_train_dataset = NYUDv2_D(data_path, transform=transform_train, transform_msk=transform_train_seg,
                                      mode='train')
        nyudv2_test_dataset = NYUDv2_D(data_path, transform=transform_test, transform_msk=transform_test_seg,
                                     mode='test')
        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(nyudv2_train_dataset)
            test_sampler = DistributedSampler(nyudv2_test_dataset)
        nice_train_loader = DataLoader(nyudv2_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(nyudv2_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)


    elif args.modality == 'hha':
        nyudv2_train_dataset = NYUDv2_HHA(data_path, transform=transform_train, transform_msk=transform_train_seg,
                                      mode='train')
        nyudv2_test_dataset = NYUDv2_HHA(data_path, transform=transform_test, transform_msk=transform_test_seg,
                                     mode='test')
        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(nyudv2_train_dataset)
            test_sampler = DistributedSampler(nyudv2_test_dataset)
        nice_train_loader = DataLoader(nyudv2_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(nyudv2_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)

    elif args.modality == 'rgbhha':
        nyudv2_train_dataset = NYUDv2_RGBHHA(data_path, transform=transform_train, transform_msk=transform_train_seg,
                                      mode='train' )
        nyudv2_test_dataset = NYUDv2_RGBHHA(data_path, transform=transform_test, transform_msk=transform_test_seg,
                                     mode='test')
        train_sampler, test_sampler = None, None
        if args.ddp:
            train_sampler = DistributedSampler(nyudv2_train_dataset)
            test_sampler = DistributedSampler(nyudv2_test_dataset)
        nice_train_loader = DataLoader(nyudv2_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=0,
                                       pin_memory=True, generator=g)
        nice_test_loader = DataLoader(nyudv2_test_dataset, batch_size=1, sampler=test_sampler, num_workers=0,
                                      pin_memory=True, generator=g)

    elif args.modality == 'taskonnomy_10':
        raise NotImplementedError

    else:
        raise NotImplementedError

    return nice_train_loader, nice_test_loader, train_sampler, test_sampler


class ZJU_RGBP(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False):
        # mean/std statistics  of each modality channel
        self.mean = torch.Tensor([141.1251, 141.37535, 138.99559, -0.01452719, -0.014152956, -0.016532706, 0.15173349, 0.1489902, 0.17018148])[:,None,None]
        self.std = torch.Tensor([139.9553, 140.5757, 141.13524, 0.38503787, 0.38248295, 0.3834839, 0.13344945, 0.13081141, 0.14260896])[:,None,None]
        self.gt_list_path = os.path.join(data_path, mode, 'label_sam')
        self.name_list = []
        self.gt_list = []

        for file_name in os.listdir(self.gt_list_path):
            img_path = [os.path.join(data_path, mode, angle, file_name.replace('.png', '_' + angle + '°.png')) for angle
                        in ['0', '45', '90', '135']]
            self.name_list.append(img_path)
            self.gt_list.append(os.path.join(self.gt_list_path, file_name))
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def raw2rgbp(self, images):
        I1, I2, I3, I4 = images
        I = 0.5 * (I1 + I2 + I3 + I4) + 1e-4
        Q = I1 - I3
        U = I2 - I4
        Q[Q == 0] = 0.0001
        I[I == 0] = 0.0001
        DoLP = np.sqrt(np.square(Q) + np.square(U)) / I
        DoLP[DoLP > 1] = 1
        AoLP = 0.5 * np.arctan(U / Q)
        # normalized_I = I / 255/ 2
        # normalized_AoLP = (AoLP+np.pi/4)/(np.pi/2)
        # normalized_DoLP = DoLP
        # rgbp = np.concatenate([normalized_I, normalized_AoLP, normalized_DoLP], axis=-1)
        rgbp = np.concatenate([I, AoLP, DoLP], axis=-1)
        return rgbp

    def __getitem__(self, index):
        """Get the images"""
        img_path = self.name_list[index]
        gt_path = self.gt_list[index]

        raw = [np.array(Image.open(angle_path), dtype=np.float32) for angle_path in img_path]
        img = self.raw2rgbp(raw)
        gt = Image.open(gt_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize, Image.NEAREST)

        CNT_THRESHOLD = 10000
        if self.mode == 'train':
            mask = np.array(mask)
            mask_values, counts = np.unique(mask, return_counts=True)
            region_ids = np.delete(mask_values, np.where(counts < CNT_THRESHOLD))[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(np.array(mask), 1, region_id)[None]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            if self.prompt == 'click':
                for region_id, region_cnt in zip(*np.unique(mask, return_counts=True)):
                    if region_id > 0 and region_cnt > CNT_THRESHOLD:
                        pt.append(random_click(np.array(mask), 1, region_id, middle=True)[None])
                        point_label.append(1)
                        valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)  # [num_click, 2]
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
        if self.transform_msk:
            mask = self.transform_msk(mask)

        name = gt_path.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        if self.mode == 'train':
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1

        # normalization
        norm_img = (img - self.mean) / self.std

        # new_modality = torch.from_numpy(np.array(raw)).permute(0,3,1,2)
        # new_modality = torch.nn.functional.interpolate(new_modality, (256,256))

        return {
            'orig_img': img[:3].flip(0),
            'new_modality': (norm_img[3:6]/4+0.5),
            'image': norm_img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }



class PGSNet_RGBP(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False):
        # mean/std statistics  of each modality channel
        self.mean = torch.Tensor([0.20545432, 0.21862635, 0.15604725, 0.47199312, 0.4637443, 0.48603472, 0.102636375, 0.10431647, 0.11925355])[:,None,None]
        self.std = torch.Tensor([0.19210099, 0.20711783, 0.18405455, 0.28202084, 0.28334478, 0.25937688, 0.11682395, 0.118849464, 0.12313089])[:,None,None]
        self.all_name_list = sorted(glob(data_path + "/{}/image/*tiff".format(mode)))
        # ignore broken
        broken_file = ['20210808013065', '20210811002205']
        self.name_list = [i for i in self.all_name_list if not any([i.__contains__(broken) for broken in broken_file])]

        # semantic
        # self.gt_list = [x.replace("image", "mask").replace("rgb.tiff", "mask.png") for x in self.name_list]
        # instance
        self.gt_list = [x.replace("image", "instance_mask").replace("rgb.tiff", "instance_mask.png") for x in self.name_list]

        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def raw2rgbp(self, img_path):
        rgb_path = img_path
        tiff_image = Image.open(rgb_path)
        rgb_image_array = np.array(tiff_image)

        aolp_path = img_path.replace("image","aolp").replace("rgb","aolp")
        tiff_image = Image.open(aolp_path)
        aolp_image_array = np.array(tiff_image)

        dolp_path = img_path.replace("image","dolp").replace("rgb","dolp")
        tiff_image = Image.open(dolp_path)
        dolp_image_array = np.array(tiff_image)

        rgbp = np.concatenate([rgb_image_array, aolp_image_array, dolp_image_array], axis=-1)

        return rgbp

    def __getitem__(self, index):
        """Get the images"""
        img_path = self.name_list[index]
        gt_path = self.gt_list[index]

        img = self.raw2rgbp(img_path)
        gt = Image.open(gt_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize, Image.NEAREST)

        CNT_THRESHOLD = 100
        if self.mode == 'train':
            mask = np.array(mask)
            mask_values, counts = np.unique(mask, return_counts=True)
            region_ids = np.delete(mask_values, np.where(counts < CNT_THRESHOLD))[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(np.array(mask), 1, region_id)[None]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            if self.prompt == 'click':
                for region_id, region_cnt in zip(*np.unique(mask, return_counts=True)):
                    if region_id > 0 and region_cnt > CNT_THRESHOLD:
                        pt.append(random_click(np.array(mask), 1, region_id, middle=True)[None])
                        point_label.append(1)
                        valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)  # [num_click, 2]
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
        if self.transform_msk:
            mask = self.transform_msk(mask)

        name = gt_path.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        if self.mode == 'train':
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1

        # normalization
        norm_img = (img - self.mean) / self.std
        return {
            'orig_img': img[:3]*255,
            'new_modality': img[3:4]*255,
            'image': norm_img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }

class PGSNet_P(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False):
        # mean/std statistics  of each modality channel
        self.mean = torch.Tensor([0.47199312, 0.4637443, 0.48603472, 0.102636375, 0.10431647, 0.11925355])[:,None,None]
        self.std = torch.Tensor([0.28202084, 0.28334478, 0.25937688, 0.11682395, 0.118849464, 0.12313089])[:,None,None]
        self.all_name_list = sorted(glob(data_path + "/{}/image/*tiff".format(mode)))
        # ignore broken
        broken_file = ['20210808013065', '20210811002205']
        self.name_list = [i for i in self.all_name_list if not any([i.__contains__(broken) for broken in broken_file])]

        # semantic
        # self.gt_list = [x.replace("image", "mask").replace("rgb.tiff", "mask.png") for x in self.name_list]
        # instance
        self.gt_list = [x.replace("image", "instance_mask").replace("rgb.tiff", "instance_mask.png") for x in self.name_list]

        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk
        self.img_id = 0

    def __len__(self):
        return len(self.name_list)

    def raw2p(self, img_path):
        rgb_path = img_path
        tiff_image = Image.open(rgb_path)
        rgb_image_array = np.array(tiff_image)

        aolp_path = img_path.replace("image","aolp").replace("rgb","aolp")
        tiff_image = Image.open(aolp_path)
        aolp_image_array = np.array(tiff_image)

        dolp_path = img_path.replace("image","dolp").replace("rgb","dolp")
        tiff_image = Image.open(dolp_path)
        dolp_image_array = np.array(tiff_image)

        rgbp = np.concatenate([aolp_image_array, dolp_image_array], axis=-1)

        return rgbp

    def raw2rgb(self, img_path):
        rgb_path = img_path
        tiff_image = Image.open(rgb_path)
        rgb_image_array = np.array(tiff_image)
        return rgb_image_array

    def __getitem__(self, index):
        """Get the images"""
        img_path = self.name_list[index]
        gt_path = self.gt_list[index]

        img = self.raw2p(img_path)
        gt = Image.open(gt_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize, Image.NEAREST)

        CNT_THRESHOLD = 100
        if self.mode == 'train':
            mask = np.array(mask)
            mask_values, counts = np.unique(mask, return_counts=True)
            region_ids = np.delete(mask_values, np.where(counts < CNT_THRESHOLD))[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(np.array(mask), 1, region_id)[None]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            if self.prompt == 'click':
                for region_id, region_cnt in zip(*np.unique(mask, return_counts=True)):
                    if region_id > 0 and region_cnt > CNT_THRESHOLD:
                        pt.append(random_click(np.array(mask), 1, region_id, middle=True)[None])
                        point_label.append(1)
                        valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)  # [num_click, 2]
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
        if self.transform_msk:
            mask = self.transform_msk(mask)

        name = gt_path.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        if self.mode == 'train':
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1

        # normalization
        img = (img - self.mean) / self.std

        # =========================================
        # rgb_img = (self.transform(self.raw2rgb(img_path)) * 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        # rgb_img = Image.fromarray(rgb_img)
        # rgb_img.save('polarization_projected/%03d_rgb.png' % (self.img_id))
        # self.img_id += 1
        # ===========================================

        return {
            'orig_img': img[:3]*255,
            'new_modality': img[3:4]*255,
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }


class ZJU_RGBP_RGB(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False):
        # mean/std statistics  of each modality channel
        self.mean = torch.Tensor([141.1251, 141.37535, 138.99559])[:,None,None]
        self.std = torch.Tensor([139.9553, 140.5757, 141.13524])[:,None,None]
        self.gt_list_path = os.path.join(data_path, mode, 'label_sam')
        self.name_list = []
        self.gt_list = []

        for file_name in os.listdir(self.gt_list_path):
            img_path = [os.path.join(data_path, mode, angle, file_name.replace('.png', '_' + angle + '°.png')) for angle
                        in ['0', '45', '90', '135']]
            self.name_list.append(img_path)
            self.gt_list.append(os.path.join(self.gt_list_path, file_name))
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def raw2rgb(self, images):
        I1, I2, I3, I4 = images
        I = 0.5 * (I1 + I2 + I3 + I4) + 1e-4
        Q = I1 - I3
        U = I2 - I4
        Q[Q == 0] = 0.0001
        I[I == 0] = 0.0001
        return I

    def __getitem__(self, index):
        """Get the images"""
        img_path = self.name_list[index]
        gt_path = self.gt_list[index]

        img = [np.array(Image.open(angle_path), dtype=np.float32) for angle_path in img_path]
        img = self.raw2rgb(img)
        gt = Image.open(gt_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize, Image.NEAREST)

        CNT_THRESHOLD = 10000
        if self.mode == 'train':
            mask = np.array(mask)
            mask_values, counts = np.unique(mask, return_counts=True)
            region_ids = np.delete(mask_values, np.where(counts < CNT_THRESHOLD))[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(np.array(mask), 1, region_id)[None]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            if self.prompt == 'click':
                for region_id, region_cnt in zip(*np.unique(mask, return_counts=True)):
                    if region_id > 0 and region_cnt > CNT_THRESHOLD:
                        pt.append(random_click(np.array(mask), 1, region_id, middle=True)[None])
                        point_label.append(1)
                        valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)  # [num_click, 2]
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
        if self.transform_msk:
            mask = self.transform_msk(mask)

        name = gt_path.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        if self.mode == 'train':
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1

        # normalization
        img = (img - self.mean)/self.std
        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }



class NYUDv2_RGBD(Dataset):
    def __init__(self, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False,
                 samples_per=1):
        self.rgbd_mean = torch.Tensor([123.675/255, 116.28/255, 103.53/255, (123.675+116.28+103.53)/3/255])[:,None,None]
        self.rgbd_std = torch.Tensor([58.395/255, 57.12/255, 57.375/255, (58.395+57.12+57.375)/3/255])[:,None,None]
        self._split_name = mode
        self.prompt = prompt
        self._data_path = data_path
        self._img_path = os.path.join(self._data_path, 'RGB')
        self._gt_path = os.path.join(self._data_path, 'Mask_gt')
        self._hha_path = os.path.join(self._data_path, 'HHA')
        self._depth_path = os.path.join(self._data_path, 'Depth')
        self.transform = transform
        self.transform_msk = transform_msk
        self._file_names = os.path.join(self._data_path, self._split_name + '.txt')
        self._name_list = []
        f = open(self._file_names, "r")
        lines = f.readlines()
        for line in lines:
            self._name_list.append(line.split('\t')[0])
        len_dataset = int(len(self._name_list) * samples_per)
        self._name_list = self._name_list[:len_dataset]
        self.img_size = 1024

    def __getitem__(self, index):
        rgb_name = self._name_list[index]
        if rgb_name in ['RGB/1133.jpg', 'RGB/398.jpg']:  # broken files
            rgb_name = 'RGB/1134.jpg'
        label_name = rgb_name.replace('RGB', 'Mask_gt').replace('jpg', 'png')
        color_label_name = rgb_name.replace('RGB', 'ColoredLabel').replace('jpg', 'png')
        hha_name = rgb_name.replace('RGB', 'HHA')
        depth_name = rgb_name.replace('RGB', 'Depth').replace('jpg', 'npy')
        img_path = os.path.join(self._data_path, rgb_name)
        color_label_path = os.path.join(self._data_path, color_label_name)
        gt_path = os.path.join(self._data_path, label_name)
        hha_path = os.path.join(self._data_path, hha_name)
        depth_path = os.path.join(self._data_path, depth_name)


        img_rgb = Image.open(img_path).convert('RGB')

        img_d = np.load(depth_path, allow_pickle=True)
        img_d = (img_d - np.min(img_d)) / (np.max(img_d) - np.min(img_d))
        img_d = img_d * 255
        img_d = img_d.astype(np.uint8)
        img = np.concatenate((np.array(img_rgb), img_d[:,:,None]), axis=-1)

        gt = Image.open(gt_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize, Image.NEAREST)

        if self._split_name == 'train':
            region_ids = np.unique(mask)[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(np.array(mask), 1, region_id)[np.newaxis, :]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            if self.prompt == 'click':
                for region_id in np.unique(mask):
                    if region_id > 0:
                        pt.append(random_click(np.array(mask), 1, region_id, middle=True)[None, :])
                        point_label.append(1)
                        valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
        if self.transform_msk:
            mask = self.transform_msk(mask)

        name = rgb_name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        norm_img = (img - self.rgbd_mean)/self.rgbd_std


        if self._split_name == 'train':
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1
        return {
            'orig_img': img[:3],
            'new_modality': img[3:],
            'image': norm_img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }

    def _fetch_data(self, img_path, gt_path, hha_path, dtype=None):
        img = self._open_image(img_path)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
        hha = self._open_image(hha_path)
        return img, gt, hha

    def __len__(self):
        return len(self._name_list)

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @classmethod
    def get_class_names(*args):
        return ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
                'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
                'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
                'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']
    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name



class NYUDv2_D(Dataset):
    def __init__(self, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False,
                 samples_per=1,if_hha=False):
        self.d_mean = torch.Tensor([(123.675+116.28+103.53)/3/255])[:,None,None]
        self.d_std = torch.Tensor([(58.395+57.12+57.375)/3/255])[:,None,None]
        self._split_name = mode
        self.prompt = prompt
        self._data_path = data_path
        self._img_path = os.path.join(self._data_path, 'RGB')
        self._gt_path = os.path.join(self._data_path, 'Mask_gt')
        self._hha_path = os.path.join(self._data_path, 'HHA')
        self._depth_path = os.path.join(self._data_path, 'Depth')
        self.transform = transform
        self.transform_msk = transform_msk
        self._file_names = os.path.join(self._data_path, self._split_name + '.txt')
        self._name_list = []
        f = open(self._file_names, "r")
        lines = f.readlines()
        for line in lines:
            self._name_list.append(line.split('\t')[0])
        len_dataset = int(len(self._name_list) * samples_per)
        self._name_list = self._name_list[:len_dataset]
        self.img_size = 1024
        self.if_hha=if_hha

    def __getitem__(self, index):
        rgb_name = self._name_list[index]
        if rgb_name in ['RGB/1133.jpg', 'RGB/398.jpg']:  # broken files
            rgb_name = 'RGB/1134.jpg'
        label_name = rgb_name.replace('RGB', 'Mask_gt').replace('jpg', 'png')
        color_label_name = rgb_name.replace('RGB', 'ColoredLabel').replace('jpg', 'png')
        hha_name = rgb_name.replace('RGB', 'HHA')
        depth_name = rgb_name.replace('RGB', 'Depth').replace('jpg', 'npy')
        img_path = os.path.join(self._data_path, rgb_name)
        color_label_path = os.path.join(self._data_path, color_label_name)
        gt_path = os.path.join(self._data_path, label_name)
        hha_path = os.path.join(self._data_path, hha_name)
        depth_path = os.path.join(self._data_path, depth_name)

        img_d = np.load(depth_path, allow_pickle=True)
        img_d = (img_d - np.min(img_d)) / (np.max(img_d) - np.min(img_d))
        img_d = img_d * 255
        img_d = img_d.astype(np.uint8)
        img = img_d[:,:,None]

        gt = Image.open(gt_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize, Image.NEAREST)

        if self._split_name == 'train':
            region_ids = np.unique(mask)[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(np.array(mask), 1, region_id)[np.newaxis, :]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            if self.prompt == 'click':
                for region_id in np.unique(mask):
                    if region_id > 0:
                        pt.append(random_click(np.array(mask), 1, region_id, middle=True)[None, :])
                        point_label.append(1)
                        valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
        if self.transform_msk:
            mask = self.transform_msk(mask)

        name = rgb_name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        norm_img = (img - self.d_mean)/self.d_std


        if self._split_name == 'train':
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1
        return {
            'orig_img': img.repeat(3,1,1),
            'new_modality': img,
            'image': norm_img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }

    def _fetch_data(self, img_path, gt_path, hha_path, dtype=None):
        img = self._open_image(img_path)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
        hha = self._open_image(hha_path)
        return img, gt, hha

    def __len__(self):
        return len(self._name_list)

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @classmethod
    def get_class_names(*args):
        return ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
                'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
                'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
                'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']
    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name

class NYUDv2_RGBHHA(Dataset):
    def __init__(self, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False,
                 samples_per=1):
        self.rgbhha_mean = torch.Tensor([123.675/255, 116.28/255, 103.53/255, 123.675/255, 116.28/255, 103.53/255])[:,None,None]
        self.rgbhha_std = torch.Tensor([58.395/255, 57.12/255, 57.375/255, 58.395/255, 57.12/255, 57.375/255])[:,None,None]
        self._split_name = mode
        self.prompt = prompt
        self._data_path = data_path
        self._img_path = os.path.join(self._data_path, 'RGB')
        self._gt_path = os.path.join(self._data_path, 'Mask_gt')
        self._hha_path = os.path.join(self._data_path, 'HHA')
        self._depth_path = os.path.join(self._data_path, 'Depth')
        self.transform = transform
        self.transform_msk = transform_msk
        self._file_names = os.path.join(self._data_path, self._split_name + '.txt')
        self._name_list = []
        f = open(self._file_names, "r")
        lines = f.readlines()
        for line in lines:
            self._name_list.append(line.split('\t')[0])
        # random.shuffle(self._name_list)
        len_dataset = int(len(self._name_list) * samples_per)
        self._name_list = self._name_list[:len_dataset]
        self.img_size = 1024

    def __getitem__(self, index):
        rgb_name = self._name_list[index]
        if rgb_name in ['RGB/1133.jpg', 'RGB/398.jpg']:  # broken files
            rgb_name = 'RGB/1134.jpg'
        label_name = rgb_name.replace('RGB', 'Mask_gt').replace('jpg', 'png')
        color_label_name = rgb_name.replace('RGB', 'ColoredLabel').replace('jpg', 'png')
        hha_name = rgb_name.replace('RGB', 'HHA')
        depth_name = rgb_name.replace('RGB', 'Depth').replace('jpg', 'npy')
        img_path = os.path.join(self._data_path, rgb_name)
        color_label_path = os.path.join(self._data_path, color_label_name)
        gt_path = os.path.join(self._data_path, label_name)
        hha_path = os.path.join(self._data_path, hha_name)
        depth_path = os.path.join(self._data_path, depth_name)

        # img = self._open_image(img_path, cv2.COLOR_BGR2RGB)
        # gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)

        img_rgb = Image.open(img_path).convert('RGB')
        # img = Image.open(color_label_path).convert('RGB')
        img_hha = Image.open(hha_path).convert('RGB')

        img = np.concatenate((np.array(img_rgb), np.array(img_hha)), axis=-1)
        gt = Image.open(gt_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize, Image.NEAREST)

        if self._split_name == 'train':
            region_ids = np.unique(mask)[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(np.array(mask), 1, region_id)[np.newaxis, :]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            if self.prompt == 'click':
                for region_id in np.unique(mask):
                    if region_id > 0:
                        pt.append(random_click(np.array(mask), 1, region_id, middle=True)[None, :])
                        point_label.append(1)
                        valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
        if self.transform_msk:
            mask = self.transform_msk(mask)

        name = rgb_name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        norm_img = (img - self.rgbhha_mean)/self.rgbhha_std


        if self._split_name == 'train':
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1
        return {
            'orig_img': img[:3],
            'new_modality': img[3:],
            'image': norm_img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }

    def _fetch_data(self, img_path, gt_path, hha_path, dtype=None):
        img = self._open_image(img_path)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
        hha = self._open_image(hha_path)
        return img, gt, hha

    def __len__(self):
        return len(self._name_list)

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @classmethod
    def get_class_names(*args):
        return ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
                'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
                'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
                'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']
    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name


class NYUDv2_HHA(Dataset):
    def __init__(self, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False,
                 samples_per=1):
        self.hha_mean = torch.Tensor([123.675/255, 116.28/255, 103.53/255])[:,None,None]
        self.hha_std = torch.Tensor([58.395/255, 57.12/255, 57.375/255])[:,None,None]
        self._split_name = mode
        self.prompt = prompt
        self._data_path = data_path
        self._img_path = os.path.join(self._data_path, 'RGB')
        self._gt_path = os.path.join(self._data_path, 'Mask_gt')
        self._hha_path = os.path.join(self._data_path, 'HHA')
        self._depth_path = os.path.join(self._data_path, 'Depth')
        self.transform = transform
        self.transform_msk = transform_msk
        self._file_names = os.path.join(self._data_path, self._split_name + '.txt')
        self._name_list = []
        f = open(self._file_names, "r")
        lines = f.readlines()
        for line in lines:
            self._name_list.append(line.split('\t')[0])
        # random.shuffle(self._name_list)
        len_dataset = int(len(self._name_list) * samples_per)
        self._name_list = self._name_list[:len_dataset]
        self.img_size = 1024

    def __getitem__(self, index):
        rgb_name = self._name_list[index]
        if rgb_name in ['RGB/1133.jpg', 'RGB/398.jpg']:  # broken files
            rgb_name = 'RGB/1134.jpg'
        label_name = rgb_name.replace('RGB', 'Mask_gt').replace('jpg', 'png')
        color_label_name = rgb_name.replace('RGB', 'ColoredLabel').replace('jpg', 'png')
        hha_name = rgb_name.replace('RGB', 'HHA')
        depth_name = rgb_name.replace('RGB', 'Depth').replace('jpg', 'npy')
        img_path = os.path.join(self._data_path, rgb_name)
        color_label_path = os.path.join(self._data_path, color_label_name)
        gt_path = os.path.join(self._data_path, label_name)
        hha_path = os.path.join(self._data_path, hha_name)
        depth_path = os.path.join(self._data_path, depth_name)

        # img = self._open_image(img_path, cv2.COLOR_BGR2RGB)
        # gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)

        img_rgb = Image.open(img_path).convert('RGB')
        # img = Image.open(color_label_path).convert('RGB')
        img_hha = Image.open(hha_path).convert('RGB')

        img = img_hha
        gt = Image.open(gt_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize, Image.NEAREST)

        if self._split_name == 'train':
            region_ids = np.unique(mask)[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(np.array(mask), 1, region_id)[np.newaxis, :]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            if self.prompt == 'click':
                for region_id in np.unique(mask):
                    if region_id > 0:
                        pt.append(random_click(np.array(mask), 1, region_id, middle=True)[None, :])
                        point_label.append(1)
                        valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
        if self.transform_msk:
            mask = self.transform_msk(mask)

        name = rgb_name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj': name}

        norm_img = (img - self.hha_mean)/self.hha_std


        if self._split_name == 'train':
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1
        return {
            'orig_img': img[:3],
            'new_modality': torch.from_numpy(np.array(img_hha)).permute(2,0,1)/255,
            'image': norm_img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }

    def _fetch_data(self, img_path, gt_path, hha_path, dtype=None):
        img = self._open_image(img_path)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
        hha = self._open_image(hha_path)
        return img, gt, hha

    def __len__(self):
        return len(self._name_list)

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @classmethod
    def get_class_names(*args):
        return ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
                'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
                'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
                'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']
    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name




class NIR(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False,
                 samples_per=1):
        self.mean = torch.Tensor([0.5038126])[:,None,None]
        self.std = torch.Tensor([0.20589863])[:,None,None]

        df = pd.read_csv(os.path.join(data_path, 'IVRG-nir_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self._split_name = mode
        self.img_id = 0

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, "Outdoor_dataset", name)

        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, "Outdoor_dataset", mask_name)

        rgb_img_path = img_path.replace("NIRimages", "RGBimages").replace("_nir", "_rgb")
        rgb_img = Image.open(rgb_img_path)
        nir_img = Image.open(img_path).convert('L')

        img_arr =np.array(nir_img)
        img = Image.fromarray(img_arr)
        gt = Image.open(msk_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize, Image.NEAREST)

        if self._split_name == 'train':
            mask = np.array(mask)
            mask_values, counts = np.unique(mask, return_counts=True)
            region_ids = np.delete(mask_values, np.where(counts < 1000))[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(mask, 1, region_id)[np.newaxis, :]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            mask = np.array(mask)
            if self.prompt == 'click':
                for region_id, region_cnt in zip(*np.unique(mask, return_counts=True)):
                    if region_id > 0 and region_cnt > 1000:
                        pt.append(random_click(mask, 1, region_id, middle=True)[None])
                        point_label.append(1)
                        valid_region_ids.append(region_id)
            pt = np.concatenate(pt, axis=0)
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
        if self.transform_msk:
            mask = self.transform_msk(mask)

        name = img_path.split('/')[-1].split(".JPG")[0]
        image_meta_dict = {'filename_or_obj': name}

        if self.mode == 'train':
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1

        norm_img = (img - self.mean)/self.std

        # ==================================================
        # rgb_img = (self.transform(rgb_img)*255).permute(1,2,0).type(torch.uint8).cpu().numpy()
        # rgb_img = Image.fromarray(rgb_img)
        # rgb_img.save('nir_projected/%03d_rgb.png' % (self.img_id))
        # self.img_id += 1
        # ==================================================


        return {
            'orig_img': img.repeat(3,1,1),
            'new_modality': img,
            'image': norm_img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }

    def __len__(self):
        return len(self.name_list)

class RGBNIR(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False,
                 samples_per=1):
        self.mean = torch.Tensor([0.43030727, 0.43242943, 0.41996416, 0.5038126])[:,None,None]
        self.std = torch.Tensor([0.25075516, 0.24733004, 0.26631436, 0.20589863])[:,None,None]

        df = pd.read_csv(os.path.join(data_path, 'IVRG-nir_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self._split_name = mode

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, "Outdoor_dataset", name)

        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, "Outdoor_dataset", mask_name)

        rgb_img_path = img_path.replace("NIRimages", "RGBimages").replace("_nir", "_rgb")
        rgb_img = Image.open(rgb_img_path)
        nir_img = Image.open(img_path).convert('L')


        img_arr = np.concatenate([np.array(rgb_img), np.array(nir_img)[:, :, np.newaxis]], axis = -1)
        img = Image.fromarray(img_arr)
        gt = Image.open(msk_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize, Image.NEAREST)

        if self._split_name == 'train':
            mask = np.array(mask)
            mask_values, counts = np.unique(mask, return_counts=True)
            region_ids = np.delete(mask_values, np.where(counts < 1000))[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(mask, 1, region_id)[np.newaxis, :]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            mask = np.array(mask)
            if self.prompt == 'click':
                for region_id, region_cnt in zip(*np.unique(mask, return_counts=True)):
                    if region_id > 0 and region_cnt > 1000:
                        pt.append(random_click(mask, 1, region_id, middle=True)[None])
                        point_label.append(1)
                        valid_region_ids.append(region_id)
            pt = np.concatenate(pt, axis=0)
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
        if self.transform_msk:
            mask = self.transform_msk(mask)

        name = img_path.split('/')[-1].split(".JPG")[0]
        image_meta_dict = {'filename_or_obj': name}

        if self.mode == 'train':
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1

        new_modality = img[3:]
        img = (img - self.mean)/self.std

        return {
            'orig_img': img[:3],
            'new_modality': new_modality,
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }

    def __len__(self):
        return len(self.name_list)



class RGBThermal(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False,
                 samples_per=1):
        df = pd.read_csv(os.path.join(data_path, 'RGB_T_Glass_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.mean = torch.Tensor([0.37960503, 0.39143142, 0.3845801, 0.35632825])[:,None,None]
        self.std = torch.Tensor([0.22783901, 0.2329615, 0.2352449, 0.21658015])[:,None,None]
        self._split_name = mode

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        rgb_img_path =name

        mask_name = self.label_list[index]
        mask_path = mask_name
        temperature_path = rgb_img_path.replace("rgb.png", "temperature.npy")

        rgb_img = Image.open(rgb_img_path)
        temperature = np.load(temperature_path)
        temperature = (temperature - np.min(temperature)) / (np.max(temperature) - np.min(temperature)) * 255
        temperature = Image.fromarray(temperature).convert('L')
        # temperature = Image.fromarray(temperature).convert('RGB')

        w, h = rgb_img.size
        masks = Image.open(mask_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = masks.resize(newsize, Image.NEAREST)

        if self._split_name == 'train':
            mask = np.array(mask)
            mask_values, counts = np.unique(mask, return_counts=True)
            region_ids = mask_values[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(mask, 1, region_id)[np.newaxis, :]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1
            mask = Image.fromarray(mask)
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            mask = np.array(mask)

            if self.prompt == 'click':
                region_ids, _ = np.unique(mask, return_counts=True)
                for region_id in region_ids[1:]:
                    pt.append(random_click(mask, 1, region_id, middle=True)[None])
                    point_label.append(1)
                    valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)
            mask = Image.fromarray(mask)

        if self.transform:
            state = torch.get_rng_state()
            rgb_img = self.transform(rgb_img)
            temperature = self.transform(temperature)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)

        name =  rgb_img_path.split('/')[-3] + '_' + rgb_img_path.split('/')[-2] + '_' + rgb_img_path.split('/')[-1].split(".png")[0]
        image_meta_dict = {'filename_or_obj': name}

        img = torch.cat((rgb_img, temperature), dim=0)
        img = (img - self.mean) / self.std

        return {
            'orig_img': img[:3],
            'new_modality': img[3:],
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }

    def __len__(self):
        return len(self.name_list)

class RGBThermal_500(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False,
                 samples_per=1):
        df = pd.read_csv(os.path.join(data_path, 'RGB_T_Glass_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        if mode == "train":
            self.name_list = self.name_list[::10]
            self.label_list = self.label_list[::10]
        else:
            self.name_list = self.name_list[::2]
            self.label_list = self.label_list[::2]

        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.mean = torch.Tensor([0.37960503, 0.39143142, 0.3845801, 0.35632825])[:,None,None]
        self.std = torch.Tensor([0.22783901, 0.2329615, 0.2352449, 0.21658015])[:,None,None]
        self._split_name = mode

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        rgb_img_path =name

        mask_name = self.label_list[index]
        mask_path = mask_name
        temperature_path = rgb_img_path.replace("rgb.png", "temperature.npy")

        rgb_img = Image.open(rgb_img_path)
        temperature = np.load(temperature_path)
        temperature = (temperature - np.min(temperature)) / (np.max(temperature) - np.min(temperature)) * 255
        temperature = Image.fromarray(temperature).convert('L')
        # temperature = Image.fromarray(temperature).convert('RGB')

        w, h = rgb_img.size
        masks = Image.open(mask_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = masks.resize(newsize, Image.NEAREST)

        if self._split_name == 'train':
            mask = np.array(mask)
            mask_values, counts = np.unique(mask, return_counts=True)
            region_ids = mask_values[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(mask, 1, region_id)[np.newaxis, :]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1
            mask = Image.fromarray(mask)
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            mask = np.array(mask)

            if self.prompt == 'click':
                region_ids, _ = np.unique(mask, return_counts=True)
                for region_id in region_ids[1:]:
                    pt.append(random_click(mask, 1, region_id, middle=True)[None])
                    point_label.append(1)
                    valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)
            mask = Image.fromarray(mask)

        if self.transform:
            state = torch.get_rng_state()
            rgb_img = self.transform(rgb_img)
            temperature = self.transform(temperature)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)

        name =  rgb_img_path.split('/')[-3] + '_' + rgb_img_path.split('/')[-2] + '_' + rgb_img_path.split('/')[-1].split(".png")[0]
        image_meta_dict = {'filename_or_obj': name}

        img = torch.cat((rgb_img, temperature), dim=0)
        img = (img - self.mean) / self.std

        return {
            'orig_img': rgb_img*255.,
            'new_modality': temperature*255.,
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }

    def __len__(self):
        return len(self.name_list)

class Thermal_500(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False,
                 samples_per=1):
        df = pd.read_csv(os.path.join(data_path, 'RGB_T_Glass_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        if mode == "train":
            self.name_list = self.name_list[::10]
            self.label_list = self.label_list[::10]
        else:
            self.name_list = self.name_list[::2]
            self.label_list = self.label_list[::2]

        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.mean = torch.Tensor([0.35632825])[:,None,None]
        self.std = torch.Tensor([ 0.21658015])[:,None,None]
        self._split_name = mode

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        rgb_img_path =name

        mask_name = self.label_list[index]
        mask_path = mask_name
        temperature_path = rgb_img_path.replace("rgb.png", "temperature.npy")

        rgb_img = Image.open(rgb_img_path)
        temperature = np.load(temperature_path)
        temperature = (temperature - np.min(temperature)) / (np.max(temperature) - np.min(temperature)) * 255
        temperature = Image.fromarray(temperature).convert('L')
        # temperature = Image.fromarray(temperature).convert('RGB')

        w, h = rgb_img.size
        masks = Image.open(mask_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = masks.resize(newsize, Image.NEAREST)

        if self._split_name == 'train':
            mask = np.array(mask)
            mask_values, counts = np.unique(mask, return_counts=True)
            region_ids = mask_values[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(mask, 1, region_id)[np.newaxis, :]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1
            mask = Image.fromarray(mask)
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            mask = np.array(mask)

            if self.prompt == 'click':
                region_ids, _ = np.unique(mask, return_counts=True)
                for region_id in region_ids[1:]:
                    pt.append(random_click(mask, 1, region_id, middle=True)[None])
                    point_label.append(1)
                    valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)
            mask = Image.fromarray(mask)

        if self.transform:
            state = torch.get_rng_state()
            rgb_img = self.transform(rgb_img)
            temperature = self.transform(temperature)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)

        name =  rgb_img_path.split('/')[-3] + '_' + rgb_img_path.split('/')[-2] + '_' + rgb_img_path.split('/')[-1].split(".png")[0]
        image_meta_dict = {'filename_or_obj': name}

        img = temperature
        img = (img - self.mean) / self.std

        return {
            'orig_img': rgb_img*255.,
            'new_modality': temperature*255.,
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }

    def __len__(self):
        return len(self.name_list)

class Thermal(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False,
                 samples_per=1):
        df = pd.read_csv(os.path.join(data_path, 'RGB_T_Glass_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.mean = torch.Tensor([0.35632825])[:,None,None]
        self.std = torch.Tensor([0.21658015])[:,None,None]
        self._split_name = mode

        self.img_id = 0

    def __getitem__(self, index):
        """Get the images"""
        filename = self.name_list[index]

        mask_name = self.label_list[index]
        mask_path = mask_name
        temperature_path = filename.replace("rgb.png", "temperature.npy")

        temperature = np.load(temperature_path)
        temperature = (temperature - np.min(temperature)) / (np.max(temperature) - np.min(temperature)) * 255
        temperature = Image.fromarray(temperature).convert('L')
        # temperature = Image.fromarray(temperature).convert('RGB')

        masks = Image.open(mask_path).convert('I')

        newsize = (self.img_size, self.img_size)
        mask = masks.resize(newsize, Image.NEAREST)

        if self._split_name == 'train':
            mask = np.array(mask)
            mask_values, counts = np.unique(mask, return_counts=True)
            region_ids = mask_values[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(mask, 1, region_id)[np.newaxis, :]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1
            mask = Image.fromarray(mask)
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            mask = np.array(mask)

            if self.prompt == 'click':
                region_ids, _ = np.unique(mask, return_counts=True)
                for region_id in region_ids[1:]:
                    pt.append(random_click(mask, 1, region_id, middle=True)[None])
                    point_label.append(1)
                    valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)
            mask = Image.fromarray(mask)

        if self.transform:
            state = torch.get_rng_state()
            temperature = self.transform(temperature)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)

        name = filename.split('/')[-3] + '_' + filename.split('/')[-2] + '_' + filename.split('/')[-1].split(".png")[0]
        image_meta_dict = {'filename_or_obj': name}

        img = temperature
        norm_img = (img - self.mean) / self.std

        # ============================================================
        # rgb_img = Image.open(filename)
        # rgb_img = (self.transform(rgb_img) * 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        # rgb_img = Image.fromarray(rgb_img)
        # rgb_img.save('thermal_projected/%03d_rgb.png' % (self.img_id))
        # self.img_id += 1
        # ==========================================================

        return {
            'orig_img': img.repeat(3,1,1),
            'new_modality': img,
            'image': norm_img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }

    def __len__(self):
        return len(self.name_list)




class ThermalFewshot(Dataset):
    def __init__(self, args, data_path, transform=None, transform_msk=None, mode='train', prompt='click', plane=False,
                 samples_per=1):
        df = pd.read_csv(os.path.join(data_path, 'RGB_T_Glass_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:, 1].tolist()
        self.label_list = df.iloc[:, 2].tolist()

        if mode == 'train':
            self.name_list = self.name_list[::100]
            self.label_list = self.label_list[::100]
        else:
            self.name_list = self.name_list[::4]
            self.label_list = self.label_list[::4]
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.mean = torch.Tensor([0.37960503, 0.39143142, 0.3845801, 0.35632825])[:,None,None]
        self.std = torch.Tensor([0.22783901, 0.2329615, 0.2352449, 0.21658015])[:,None,None]
        self._split_name = mode

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        rgb_img_path =name

        mask_name = self.label_list[index]
        mask_path = mask_name
        temperature_path = rgb_img_path.replace("rgb.png", "temperature.npy")
        
        rgb_img = Image.open(rgb_img_path)
        temperature = np.load(temperature_path)
        temperature = (temperature - np.min(temperature)) / (np.max(temperature) - np.min(temperature)) * 255
        temperature = Image.fromarray(temperature).convert('L')
        # temperature = Image.fromarray(temperature).convert('RGB')

        w, h = rgb_img.size
        masks = Image.open(mask_path).convert('I')
    
        newsize = (self.img_size, self.img_size)
        mask = masks.resize(newsize, Image.NEAREST)

        if self._split_name == 'train':
            mask = np.array(mask)
            mask_values, counts = np.unique(mask, return_counts=True)
            region_ids = mask_values[1:]
            region_id = np.random.choice(region_ids)
            pt = random_click(mask, 1, region_id)[np.newaxis, :]
            pt = np.ascontiguousarray(pt)
            point_label = np.array([1])
            valid_region_ids = np.array([region_id])
            mask[mask != region_id] = 0
            mask[mask == region_id] = 1
            mask = Image.fromarray(mask)
        else:
            pt = []
            point_label = []
            valid_region_ids = []
            mask = np.array(mask)

            if self.prompt == 'click':
                region_ids, _ = np.unique(mask, return_counts=True)
                for region_id in region_ids[1:]:
                    pt.append(random_click(mask, 1, region_id, middle=True)[None])
                    point_label.append(1)
                    valid_region_ids.append(region_id)

            pt = np.concatenate(pt, axis=0)
            point_label = np.array(point_label)
            valid_region_ids = np.array(valid_region_ids)
            mask = Image.fromarray(mask)

        if self.transform:
            state = torch.get_rng_state()
            rgb_img = self.transform(rgb_img)
            temperature = self.transform(temperature)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)

        name =  rgb_img_path.split('/')[-3] + '_' + rgb_img_path.split('/')[-2] + '_' + rgb_img_path.split('/')[-1].split(".png")[0]
        image_meta_dict = {'filename_or_obj': name}
    
        img = torch.cat((rgb_img, temperature), dim=0)
        img = (img - self.mean) / self.std

        img = img[3:]
        return {
            'image': img,
            'label': mask,
            'p_label': point_label,
            'pt': pt,
            'valid_region_ids': valid_region_ids,
            'image_meta_dict': image_meta_dict,
        }

    def __len__(self):
        return len(self.name_list)
