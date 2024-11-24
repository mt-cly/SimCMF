import sys

from torch import nn
from utils import *
from tqdm import tqdm
from monai.losses import DiceCELoss
import torch.distributed as dist
import torch


def train_sam(args, net: nn.Module, optimizer, train_loader, epoch, writer=None, schedulers=None, vis = 50):
    # train mode
    net.train()
    epoch_loss = 0
    optimizer.zero_grad()
    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    # # ========== statistics of mean & std ==============
    # values = []
    # for idx, pack in enumerate(train_loader):
    #     imgs = pack['image'].numpy()
    #     values.append(imgs)
    # print('mean', [np.stack(values[:-1])[:,:,i].mean() for i in range(values[0].shape[1])])
    # print('std' , [np.stack(values[:-1])[:,:,i].std() for i in range(values[0].shape[1])])
    # # =================================================

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for iter, pack in enumerate(train_loader):
            imgs = pack['image'].to(dtype = torch.float32).cuda()
            masks = pack['label'].to(dtype = torch.float32).cuda()

            # 'pt' points should with shape of [bs, num_click, 2]
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']
            name = ''
            for item_name in pack['image_meta_dict']['filename_or_obj']:
                name += item_name.split('/')[-1].split('.')[0] + '_'


            '''Train'''
            click_prompt = (torch.as_tensor(pt, dtype=torch.float).cuda(), torch.as_tensor(point_labels, dtype=torch.float).cuda())
            pred = net.forward(imgs, click_prompt)
            loss = lossfunc(pred, masks)
            epoch_loss += loss.item()
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            loss.backward()
            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            '''vis images'''
            if vis and (iter+1) % vis == 0:
                save_path = os.path.join(args.path_helper['sample_path'], f'train_{name}_epoch={epoch}.jpg')
                new_vis(args.modality, pack['orig_img'], pred, masks, save_path, pack['new_modality'], reverse=False, points=click_prompt[0][:, 0])
            pbar.update()

    return epoch_loss/len(train_loader)


def validation_sam(args, val_loader, epoch, net: nn.Module , vis):
     # eval mode
    net.eval()
    net = net.module if args.ddp else net

    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    tot = 0
    hard = 0
    # threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    threshold = [0]

    lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    masks_num = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype = torch.float32).cuda()
            orig_img = pack['orig_img'].to(dtype = torch.float32)
            masksw = pack['label'].to(dtype = torch.float32).cuda()
            new_modality = pack['new_modality'].to(dtype = torch.float32)
            valid_region_ids = pack['valid_region_ids']
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']
            
            buoy = 0
            evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):

                pt = ptw
                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    #true_mask_ave = cons_tensor(true_mask_ave)

                b_size,c,w,h = imgs.size()
                imgs = imgs.to(dtype =torch.float32).cuda()
                coords_torch = torch.as_tensor(pt, dtype=torch.float32).cuda()
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int).cuda()
                
                '''test'''
                with torch.no_grad():
                    for bs_id in range(b_size):
                        name_tmp = name[bs_id].split('/')[-1].split('.')[0]
                        img_tmp = imgs[bs_id:bs_id+1]
                        coords_tmp, label_tmp = coords_torch[bs_id:bs_id+1], labels_torch[bs_id:bs_id+1]
                        # first calculate image feature which can be used repeatedly
                        img_feature = net.image_encoder(img_tmp)
                        for pnt_id, region_id in enumerate(valid_region_ids[bs_id]):
                            mask_tmp = masks.detach().clone()
                            mask_tmp[mask_tmp != region_id] = 0
                            mask_tmp[mask_tmp == region_id] = 1

                            prompt_tmp = (coords_tmp[:, pnt_id: pnt_id+1], label_tmp[:, pnt_id: pnt_id+1])
                            se, de = net.prompt_encoder(
                                points=prompt_tmp,
                                boxes=None,
                                masks=None,
                            )
                            pred, _ = net.mask_decoder(
                                image_embeddings=img_feature,  # [2, 256, 64, 64]
                                image_pe=net.prompt_encoder.get_dense_pe(),  # [1, 256, 64, 64]
                                sparse_prompt_embeddings=se,  # [2,2,256]
                                dense_prompt_embeddings=de,  # [2, 256, 64, 64]
                                multimask_output=False,
                            )

                            tot += lossfunc(pred, mask_tmp)

                            '''vis images'''
                            if vis and (bs_id+1) % vis == 0:
                                save_path = os.path.join(args.path_helper['sample_path'], f'test_{name_tmp}_epoch={epoch}_maskid={region_id}.jpg')
                                new_vis(args.modality, orig_img, pred, mask_tmp, save_path, new_modality, reverse=False, points=prompt_tmp[0][:, 0])

                            temp = eval_seg(pred, mask_tmp, threshold)
                            mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
                            masks_num += 1
            pbar.update()

    tol, eiou, edice = tot.item(), mix_res[0], mix_res[1]
    # gather from all gpus if ddp
    if args.ddp:
        result = torch.Tensor([tol, eiou, edice, masks_num]).type(torch.float64).cpu()
        all_rst = [torch.zeros(4, dtype=torch.float64) for _ in range(dist.get_world_size())]
        dist.all_gather(all_rst, result)
        result = torch.stack(all_rst).sum(0)
        tol, eiou, edice, masks_num = tuple(result)
    tol, eiou, edice = tol/masks_num, eiou/masks_num, edice/masks_num
    return tol, eiou, edice



