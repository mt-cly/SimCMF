#-net: {sam_lora, sam_mlp_adapter, sam_medical_adapter, sam_prompt, sam_prefix, sam_full_finetune, sam_linear_probing}
#-proj_type: {medical_adapter, baseline_0, baseline_1, baseline_2, simmat, vipt_shallow, vipt_deep, cmx }

for net in sam_lora sam_mlp_adapter sam_medical_adapter sam_prompt sam_prefix sam_full_finetune sam_linear_probing ; do
  for proj_type in medical_adapter baseline_0 baseline_1 baseline_2 simmat vipt_shallow vipt_deep cmx; do
    python train.py -net ${net} -proj_type ${proj_type} -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth \
    -exp_name local_exp -image_size 1024 -b 1 -dataset rgbnir -data_path ./data/IVRG_RGBNIR -val_freq 1 -vis 1
  done
done
