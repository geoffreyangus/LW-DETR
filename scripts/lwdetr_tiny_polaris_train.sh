model_name='lwdetr_tiny_polaris'
polaris_path=$1

# TODO(geoff): >60 epochs is a hack; we want to resume the COCO training, which has 60 epochs.
python -u -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env \
    main.py \
    --lr 1e-4 \
    --lr_encoder 1.5e-4 \
    --batch_size 4 \
    --weight_decay 1e-4 \
    --epochs 200 \
    --lr_drop 60 \
    --lr_vit_layer_decay 0.8 \
    --lr_component_decay 0.7 \
    --encoder vit_tiny \
    --vit_encoder_num_layers 6 \
    --window_block_indexes 0 2 4 \
    --out_feature_indexes 1 3 5 \
    --dec_layers 3 \
    --group_detr 13 \
    --two_stage \
    --projector_scale P4 \
    --hidden_dim 256 \
    --sa_nheads 8 \
    --ca_nheads 16 \
    --dec_n_points 2 \
    --bbox_reparam \
    --lite_refpoint_refine \
    --num_queries 100 \
    --ia_bce_loss \
    --cls_loss_coef 1 \
    --num_select 100 \
    --dataset_file polaris \
    --polaris_path $polaris_path \
    --square_resize_div_64 \
    --output_dir output/$model_name \
    --resume /mnt/e/model_weights/LWDETR_tiny_60e_coco.pth