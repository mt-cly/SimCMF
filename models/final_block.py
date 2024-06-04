
import torch.nn as nn


def get_final_block(SAM_embedding_chans, proj_type):
    return Identity_Block()

class Identity_Block(nn.Module):
    def __init__(self):
        super(Identity_Block, self).__init__()

    def forward(self, ms_rgb_embedding, ms_x_embedding):
        return ms_rgb_embedding[-1]
