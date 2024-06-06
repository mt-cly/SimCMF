# adapt SAM to RGB+NIR modality with lora PEFT
net=sam_lora
modality=rgbnir
lr=3e-4
proj_type=simmat
exp_name=aaa

python -u train.py -net ${net} \
  -proj_type ${proj_type} \
  -exp_name ${exp_name} \
  -lr ${lr} \
  -b 4 -modality ${modality} -val_freq 5



# adapt SAM to RGB+NIR modality with full finetune
net=sam_full_finetune
dataset=rgbnir
lr=3e-5
modality=simmat
exp_name=${dataset}_projtype${proj_type}_uniformInit_${net}_lr${lr}

python -u train.py -net ${net} \
  -proj_type ${proj_type} \
  -exp_name ${exp_name} \
  -lr ${lr} \
  -b 4 -modality ${modality} -val_freq 5