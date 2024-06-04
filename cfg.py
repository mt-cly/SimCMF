import argparse

def valid_type(value):
    choices = ['baseline_0', 'baseline_1', 'baseline_2', 'simmat']
    if value in choices or value.__contains__('preconv') or value.__contains__('preattn'):
        return value
    raise 'wrong -proj_type'


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-ddp', action='store_true', default=False, help='using DDP')
    parser.add_argument('-proj_type', type=valid_type, help='the pre-projection before foundation model')
    parser.add_argument('-discard_patch', action='store_true', help='if discard the patch embedding weights')
    parser.add_argument('-freeze_patch', action='store_true', help='if freeze the patch embedding weights')
    parser.add_argument('-freeze_prompt_encoder', action='store_true', help='if freeze the prompt_encoder')
    parser.add_argument('-non_freeze_prompt_encoder', action='store_true', help='if not freeze the prompt_encoder')
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-ratio', type=float, default=1., help='evaluate data number effect')
    parser.add_argument('-seg_net', type=str, default='transunet', help='net type')
    parser.add_argument('-exp_name', type=str, required=True, help='net type')
    parser.add_argument('-type', type=str, default='map', help='condition type:ave,rand,rand_map')
    parser.add_argument('-vis', type=int, default=0, help='visualization')
    parser.add_argument('-reverse', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-pretrain', type=str, help='adversary reverse')
    parser.add_argument('-val_freq',type=int,default=100,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-sim_gpu', type=int, default=0, help='split sim to this gpu')
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-image_size', type=int, default=256, help='image_size')
    parser.add_argument('-out_size', type=int, default=256, help='output_size')
    parser.add_argument('-patch_size', type=int, default=2, help='patch_size')
    parser.add_argument('-dim', type=int, default=512, help='dim_size')
    parser.add_argument('-depth', type=int, default=1, help='depth')
    parser.add_argument('-heads', type=int, default=16, help='heads number')
    parser.add_argument('-mlp_dim', type=int, default=1024, help='mlp_dim')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-patch_embedding_lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-uinch', type=int, default=1, help='input channel of unet')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
    parser.add_argument('-weights', type=str, default = None, help='the weights file you want to test')
    parser.add_argument('-base_weights', type=str, default = 0, help='the weights baseline')
    parser.add_argument('-sim_weights', type=str, default = 0, help='the weights sim')
    parser.add_argument('-distributed', default=None, type=str, help='multi GPU ids to use')
    parser.add_argument('-dataset', default='isic' ,type=str,help='dataset name')
    parser.add_argument('-sam_ckpt', default=None , help='sam checkpoint address')
    parser.add_argument('-thd', type=bool, default=False , help='3d or not')
    parser.add_argument('-chunk', type=int, default=96 , help='crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4 , help='sample pos and neg')
    parser.add_argument('-roi_size', type=int, default=96 , help='resolution of roi')
    parser.add_argument('-evl_chunk', type=int, default=None , help='evaluation chunk')
    parser.add_argument('-samples_per', type=float, default=1 , help='percentage of training samples')
    parser.add_argument(
    '-data_path',
    type=str,
    default='../data',
    help='The path of segmentation data')
    # '../dataset/RIGA/DiscRegion'
    # '../dataset/ISIC'
    opt = parser.parse_args()

    return opt
