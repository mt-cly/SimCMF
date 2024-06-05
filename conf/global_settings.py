import os
from datetime import datetime

TIME_NOW = datetime.now().isoformat()

# total training epoches
EPOCH = 51

# tensorboard log dir
LOG_DIR = 'runs'

modality_channel_map = {'zju-rgbp': 9, 'pgsnet_rgbp': 9, 'pgsnet_p': 6, 'rgbd': 4, 'd': 1, 'rgbhha': 6, 'hha': 3,
                        'nir': 1, 'rgbnir': 4, 'rgbt': 4, 'rgbt_500': 4, 't': 1, 't_500': 1}
datasets_folder = './data'
modality_datapath_map = {'zju-rgbp': f'{datasets_folder}/zju-rgbp', 'pgsnet_rgbp': f'{datasets_folder}/pgsnet',
                         'pgsnet_p': f'{datasets_folder}/pgsnet', 'rgbd': f'{datasets_folder}/NYUDepthv2',
                         'd': f'{datasets_folder}/NYUDepthv2', 'rgbhha': f'{datasets_folder}/NYUDepthv2',
                         'hha': f'{datasets_folder}/NYUDepthv2', 'nir': f'{datasets_folder}/IVRG_RGBNIR',
                         'rgbnir': f'{datasets_folder}/IVRG_RGBNIR', 'rgbt': f'{datasets_folder}/RGB-Thermal-Glass',
                         'rgbt_500': f'{datasets_folder}/RGB-Thermal-Glass',
                         't': f'{datasets_folder}/RGB-Thermal-Glass', 't_500': f'{datasets_folder}/RGB-Thermal-Glass'}

# ========== parameter efficient tuning =================
# refer to the paper https://arxiv.org/pdf/2110.04366.pdf
# lora introduce 4 * r * d params
# mlp_adapter introduce 2 * r * d params
# prompt tuning introduce l * d params
# prefix tuning introduce  2 * l * d params
# where r is the hidden_dim, l is the number of prefix prompt
# to balance the number of tunable parameters across different PEFT, set different l values
LORA_R = 100
ADAPTER_R = 200
PROMPT_L = 400
PREFIX_L = 200
