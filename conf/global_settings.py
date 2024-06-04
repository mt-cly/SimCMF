import os
from datetime import datetime

TIME_NOW = datetime.now().isoformat()

#total training epoches
EPOCH = 51

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

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








