

# SimCMF: A Simple Cross-modal Fine-tuning Strategy from Vision Foundation Models to Any Imaging Modality
- Authors: [Chengyang Lei](https://chenyanglei.github.io/), [Liyi Chen](https://scholar.google.com/citations?user=nMev-10AAAAJ&hl=zh-CN), [Jun Cen](https://cen-jun.com/), [Xiao Chen](https://scholar.google.com/citations?user=swFOM1wAAAAJ&hl=en), [Zhen Lei](http://www.cbsr.ia.ac.cn/users/zlei/), [Felix Heide](https://www.cs.princeton.edu/~fheide/), [Qifeng Chen](https://cqf.io/), [Zhaoxiang Zhang](https://zhaoxiangzhang.net/) 

***
SimCMF aims to transfer the ability of large RGB-based models to other modalities (e.g., Depth, Thermal, Polarization), which suffering from limited training data. For example,SimCMF enable the Segment Anything Model the ability to handle modality beyond RGB images.           
<p align="center">
<img src="resources/overview.png" width="95%">
</p>


## <a name="GettingStarted"></a>Getting Started
Firstly, prepare the project and create the environment.
```
git clone https://github.com/mt-cly/SimCMF
cd SimCMF
conda create -n simcmf python=3.10
conda activate simcmf
pip install -r requirements.txt
# pretrained SAM-B 
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv sam_vit_b_01ec64.pth checkpoint/sam 
```

We provide segmentation benchmark to study the segmentation performance in various modalities.

| Dataset                    | Supporting Modalities          | Link                                                                                                                                                        |
|----------------------------|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| IVRG_RGBNIR  | NIR, NIR+RGB                   | [download(1.0G)](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118917r_connect_polyu_hk/ESDT5HdpqytGqblw7dbWcWQBEKkDHcs_vZokddOoUGtTrA?e=hvcmS4)  |
| RGB-Thermal-Glass            | Thermal, Thermal+RGB           | [download(3.0G)](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118917r_connect_polyu_hk/EZxLLauNnjJAkPHAr-_IZCEBbCl80g54ZnN8tuH5iriQdg?e=zArl7G)  |
|NYUDepthv2| Depth, HHA, Depth+RGB, HHA+RGB | [download(1.6G)](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118917r_connect_polyu_hk/ESnyZODEalVNqCdN-AcEIwUBJZs_8-CP4ABTVkcncYiSSQ?e=FUMUge)  | 
|pgsnet | AOLP+DOLP, AOLP+DOLP+RGB       | [download(15.5G)](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118917r_connect_polyu_hk/EftFWER7U_VKjyHE8CKZQnEBy4BgJCPuQVLLeoUXFfhq1g?e=wt0mgH) |
|zju-rgbp| AOLP+DOLP, AOLP+DOLP+RGB       | [download(0.3G)](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21118917r_connect_polyu_hk/EdXaBm7dwx5GnDoqnPYm8-IBTsIrwaHRTN4y-lPa2L_qDw?e=bP40CX)  |


You can download one or all benchmark from given links, unzip and move them to the `data` folder, the file structure should be as follows. 

```
--SimCMF
   |--data
     |--IVRG_RGBNIR
     |--NYUDepthv2
     |--pgsnet
     |--RGB-Thermal-Glass
     |--zju-rgbp
```

You can simply execute `python train.py` followed by optional arguments.
```python
  -net         # specify the tuning methods. Options: {sam_full_finetune, sam_linear_probing, sam_mlp_adapter, sam_lora, sam_prompt, sam_prefix}
  -modality    # modality name. Options:{pgsnet_rgbp, pgsnet_p, rgbd, d, rgbhha, hha, nir, rgbnir, rgbt, t,zju-rgbp}
  -proj_type   # the pre-projection before foundation model Options: {simcmf, baseline_a, baseline_b, baseline_c, baseline_d}
  -exp_name    # the experiment name
  -val_freq    # interval epochs between each validation. Default: 5
  -b           # batch size. Default: 4
  -lr          # learning rate. It is suggested to set 3e-4 for PEFT, 3e-5 for Full Finetuning
  -weights     # the path to trained weights you want to resume
```
If you want to use DDP, just add extra `-ddp` to the command.

We provide an example command to perform adapting SAM to NIR modality in `train.sh`.
```shell
sh train.sh
```


[//]: # (Following is an example to adapt SAM-B to modality of AOLP+DOLP+RGB with proposed SimCMF:)

[//]: # (```python)

[//]: # (python train.py -net sam_lora -modality pgsnet_rgbp -proj_type simcmf -exp_name exps -lr 3e-4 -ddp)

[//]: # (```)


## Citation

```
@misc{ikemura2024robust,
      title={SimCMF: A Simple Cross-modal Fine-tuning Strategy from Vision Foundation Models to Any Imaging Modality},
      author={Lei, Chengyang and Chen, Liyi and Cen, Jun and Chen, Xiao and Lei, Zhen and Heide, Felix and Chen, Qifeng and Zhang, Zhaoxiang},
      year={2024},
      eprint={2409.08083},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
  }
```

## Acknowledgements
The code is based on [Medical-SAM-Adapter](https://github.com/KidsWithTokens/Medical-SAM-Adapter).