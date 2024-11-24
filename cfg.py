import argparse

def valid_type(value):
    choices = ['baseline_a', 'baseline_b', 'baseline_c', 'baseline_d', 'simcmf', 'zeroshot']
    if value in choices or value.__contains__('preconv') or value.__contains__('preattn'):
        return value
    raise 'wrong -proj_type'


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-uniformInit', action="store_true", help='if uniform init')
    parser.add_argument('-seed', type=int, default=8888, help='seed')
    parser.add_argument('-net', type=str, required=True, help='specify the tuning methods from {sam_lora, sam_prefix}.')
    parser.add_argument('-modality', default='rgbp', type=str,help='modality name')
    parser.add_argument('-proj_type', type=valid_type, help='the pre-projection before foundation model')
    parser.add_argument('-exp_name', type=str, required=True, help='net type')
    parser.add_argument('-vis', type=int, default=0, help='if save visualization')
    parser.add_argument('-image_size', type=int, default=1024, help='image_size')
    parser.add_argument('-out_size', type=int, default=256, help='output_size')
    parser.add_argument('-lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('-b', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('-val_freq', type=int,default=5, help='interval epoch  between each validation')
    parser.add_argument('-weights', type=str, default=None, help='the weights file you want to resume')
    parser.add_argument('-sam_ckpt', default='./checkpoint/sam/sam_vit_b_01ec64.pth', help='sam checkpoint address')
    parser.add_argument('-ddp', action='store_true', default=False, help='if using DDP')
    opt = parser.parse_args()

    return opt
