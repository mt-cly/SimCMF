#-net: {sam_lora, sam_mlp_adapter, sam_medical_adapter, sam_prompt, sam_prefix, sam_full_finetune, sam_linear_probing}
#-proj_type: {baseline_0, baseline_1, baseline_2, simmat }

lr=3e-5
for net in sam_lora  ; do
  for proj_type in simmat ; do
    dataset=rgbd
    data_path=./data/NYUDepthv2
    python train.py -net ${net} -proj_type ${proj_type} -lr ${lr} -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
    -exp_name local_exp -image_size 1024 -b 4 -dataset ${dataset} -data_path ${data_path} -val_freq 5 -vis 0
  done
done
