import sys
import cv2
from torch.autograd import Function
import torchvision
import random
import logging
import os
import time
from datetime import datetime
from itertools import chain
import dateutil.tz
import cfg
import numpy as np
import torch
from conf.global_settings import modality_channel_map


def get_network(args, net, proj_type):
    """ return given network
    """
    in_chans = modality_channel_map[args.modality]
    pretrained_state_dict = torch.load(args.sam_ckpt) if args.sam_ckpt and len(args.sam_ckpt)>0 else None
    params = {'checkpoint': args.sam_ckpt,
              'in_chans': in_chans,
              'proj_type': proj_type,
              'pretrained_state_dict': pretrained_state_dict}

    if net in ['sam_full_finetune', 'sam_linear_probing']:
        from models.sam_naive import sam_model_registry
        net = sam_model_registry['vit_b'](args, **params)
    elif net == 'sam_mlp_adapter':
        from models.sam_mlp_adapter import sam_model_registry
        net = sam_model_registry['vit_b'](args, **params)
    elif net == 'sam_lora':
        from models.sam_lora import sam_model_registry
        net = sam_model_registry['vit_b'](args, **params)
    elif net == 'sam_prompt':
        from models.sam_prompt import sam_model_registry
        net = sam_model_registry['vit_b'](args, **params)
    elif net == 'sam_prefix':
        from models.sam_prefix import sam_model_registry
        net = sam_model_registry['vit_b'](args, **params)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net


def set_trainable_params(net, net_name):
    trainable_params_names = [k for k, v in net.named_parameters()]
    if net_name in ['sam_full_finetune']:
        pass
    elif net_name in ['sam_linear_probing']:
        for n, value in net.named_parameters():
            if all([not n.__contains__(i) for i in ["patch_embed", "pos_embed"]]):
                trainable_params_names.remove(n)
                value.requires_grad = False
    elif net_name in ['sam_lora']:
        for n, value in net.named_parameters():
            if "lora" not in n:
                trainable_params_names.remove(n)
                value.requires_grad = False
    elif net_name in ['sam_prompt']:
        for n, value in net.named_parameters():
            if "prompt" not in n:
                trainable_params_names.remove(n)
                value.requires_grad = False
    elif net_name in ['sam_prefix']:
        for n, value in net.named_parameters():
            if "prefix" not in n:
                trainable_params_names.remove(n)
                value.requires_grad = False
    elif net_name in ['sam_mlp_adapter']:
        for n, value in net.named_parameters():
            if "Adapter" not in n:
                trainable_params_names.remove(n)
                value.requires_grad = False
    else:
        raise NotImplementedError

    # ==== projector and deep_fusion_blocks always trainable =====
    for n, value in net.named_parameters():
        if "image_encoder.projector" in n:
            trainable_params_names.append(n)
            value.requires_grad = True
        elif "image_encoder.deep_fusion_blocks" in n:
            trainable_params_names.append(n)
            value.requires_grad = True
        elif "image_encoder.final_block" in n:
            trainable_params_names.append(n)
            value.requires_grad = True

    # ===== prompt encoder is always frozen, even it is set to requires_grad=True, refer to forward function of  sam.
    for n, value in net.prompt_encoder.named_parameters():
        value.requires_grad = False

    # ==============Params
    print("TOTAL NUMBER OF PARAMS: {}".format(np.array([torch.numel(i) for i in net.parameters()]).sum()))
    print("TOTAL NUMBER OF TRAINABLE PARAMS: {}".format(
        np.array([torch.numel(i) for i in net.parameters() if i.requires_grad]).sum()))
    print("=============== end trainable params =======================")

    # ==================FLOPS
    # from thop import profile
    # img = torch.Tensor(1,9,1024,1024)
    # pnt = (torch.zeros(1, 1, 2),torch.ones(1, 1))
    # flops,params = profile(net, inputs=(img,pnt,))
    # print(flops/1e9, params)

    return trainable_params_names

def resume_weights(net, weights):
    if weights is not None:
        print(f'=> resuming net weights from {weights}')
        assert os.path.exists(weights)
        checkpoint_file = os.path.join(weights)
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        # args.path_helper = checkpoint['path_helper']
        # logger = create_logger(args.path_helper['log_path'])
        # print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    return net


def resume_optim_scheduler_epoch(optimizer, scheduler, weights):
    start_epoch = 0
    if weights is not None:
        print(f'=> resuming optimizer & scheduler & epoch from {weights}')
        assert os.path.exists(weights)
        checkpoint_file = os.path.join(weights)
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

    return optimizer, scheduler, start_epoch


def save_checkpoint(states, is_best, output_dir,filename):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger('sam')
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)
    file_handler = logging.FileHandler(final_log_file)
    logger.addHandler(file_handler)

    return logger

def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict

def iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def eval_seg(pred,true_mask_p,threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
        return eiou / len(threshold), edice / len(threshold)

def random_click(mask, point_labels = 1, region_id=1, middle=False):
    ''' randomly sample the click from regions of mask == mask_id
    Args:
        mask: [h, w]
        point_labels: unused params
        region_id: the region id to be sampled
    Return:
        click: the sampled click coordination with shape [1, 2]
    '''
    indices = np.argwhere(mask == region_id)
    # !!!! to align the SAM input, return the click with order (x, y) instead of (y, x)
    indices = np.ascontiguousarray(np.flip(indices, axis=1))
    if middle:
        return indices[int(len(indices)/2)]
    else:
        return indices[np.random.randint(len(indices))]


def generate_click_prompt(img, msk, pt_label=1):
    # return: prompt, prompt mask
    pt_list = []
    msk_list = []
    b, c, h, w, d = msk.size()
    msk = msk[:,0,:,:,:]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j,:,:,i]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                # generate a random array between [0-h, 0-h]:
                random_index = torch.randint(0, h, (2,)).to(device = msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                # convert bool tensor to int
                new_s = (msk_s == label).to(dtype = torch.float)
                # new_s[msk_s == label] = 1
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)
        pts = torch.stack(pt_list_s, dim=0)
        msks = torch.stack(msk_list_s, dim=0)
        pt_list.append(pts)
        msk_list.append(msks)
    pt = torch.stack(pt_list, dim=-1)
    msk = torch.stack(msk_list, dim=-1)

    # !!!! to align the SAM input, return the click with order (x, y) instead of (y, x)
    pt = np.ascontiguousarray(np.flip(pt, axis=1))
    msk = msk.unsqueeze(1)

    return img, pt, msk #[b, 2, d], [b, c, h, w, d]

def visal_click(modality, img, click):
    '''
    img: [H, W, 3] or [H, W, 1] \in [0, 255]
    mask: [H, W, 1] \in {0,1}
    '''
    if type(img) == torch.Tensor:
        img = img.cpu().numpy()
    if type(click) == torch.Tensor:
        click = click.cpu().numpy()
    x,y  = click
    click_map = np.zeros_like(img)[...,:1]
    h, w, _ = click_map.shape
    d = 5
    for delta_x in range(-d-2, d+2):
        for delta_y in range(-d-2, d+2):
            if delta_y* delta_y+ delta_x* delta_x > d*d:
                continue
            if y+delta_y < 0 or y+delta_y >=h:
                continue
            if x+delta_x < 0 or x+delta_x >= w:
                continue
            click_map[y+delta_y, x+delta_x, 0] = 1
    map = visual_mask(modality, img, click_map)
    return map


def visual_mask(modality, img, mask):
    '''
    img: [H, W, 3] or [H, W, 1] \in [0, 255]
    mask: [H, W, 1] \in {0,1}
    '''
    palette_dict = {'hha':[255, 255, 255],
                    'rgbhha': [255, 255, 255],
                    'd': [230, 170, 143],
                    'rgbd': [230, 170, 143],
                    'nir': [0, 255, 255],
                    'rgbnir': [0, 255, 255],
                    }
    palette = np.array(palette_dict[modality])
    if type(img) == torch.Tensor:
        img = img.cpu().numpy()
    if type(mask) == torch.Tensor:
        mask = mask.cpu().numpy()
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, -1)
    alpha = 0.4
    color = np.ones_like(mask) * palette[None, None]
    map = img * (mask==0) + (img * alpha + color * (1-alpha)) * (mask == 1)
    return map


def new_vis(modality, imgs, pred_masks, gt_masks, save_path, new_modality, reverse = False, points = None):
    """
    params:
        imgs: [bs, c, h, w], where c is the channel of modality. for example, 3 for rgb, 9 for polarization
        pred_masks: [bs, c, h, w], where c=1/2
        gt_masks: [bs, c, h, w], where c=1/2
        save_path: str
        reverse: if true, setting pred_masks=1-pred_masks, gt_masks=1-gt_masks
        points: [bs, 2]
    """
    b,c,h,w = pred_masks.size()
    imgs = torchvision.transforms.Resize((h, w))(imgs)
    img = imgs.permute(0,2,3,1)[0]
    new_modality = torchvision.transforms.Resize((h, w))(new_modality)
    new_modality_vis = new_modality.permute(0,2,3,1)[0].cpu().numpy()*255
    if modality == 'd':
        new_modality_vis = cv2.applyColorMap((new_modality_vis).astype(np.uint8), 4)
    elif modality == 'hha':
        new_modality_vis = np.flip(new_modality_vis, -1)
    # new_modality_vis = (new_modality_vis*255).astype(np.uint8)
    pred_mask = pred_masks.permute(0,2,3,1)[0]
    gt_mask = gt_masks.permute(0,2,3,1)[0]
    vis_pred = visual_mask(modality, new_modality_vis , pred_mask>0)
    vis_gt = visual_mask(modality, new_modality_vis, gt_mask)
    click = np.round(points.cpu()/4).to(dtype = torch.int)[0] # 1024 in 256 out
    vis_click = visal_click(modality, new_modality_vis, click)
    cv2.imwrite(save_path.replace('.jpg', '_pred.jpg'), vis_pred)
    cv2.imwrite(save_path.replace('.jpg', '_gt.jpg'), vis_gt)
    cv2.imwrite(save_path.replace('.jpg', '_gt_mask.jpg'), 255 * gt_mask.cpu().numpy())    
    cv2.imwrite(save_path.replace('.jpg', '_pred_mask.jpg'),255 * (pred_mask>0).long().cpu().numpy())
    cv2.imwrite(save_path.replace('.jpg', '_rgb.jpg'), img.cpu().numpy())
    cv2.imwrite(save_path.replace('.jpg', '_new_modality.jpg'), new_modality_vis)
    cv2.imwrite(save_path.replace('.jpg', '_click.jpg'), vis_click)
    #
    # new_modality_vis = new_modality_vis.cpu().numpy()
    # cv2.imwrite(save_path.replace('.jpg', '_newmodality.jpg'), new_modality_vis)
