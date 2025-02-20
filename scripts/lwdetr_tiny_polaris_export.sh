model_name='lwdetr_tiny_polaris'
polaris_path=$1
checkpoint=$2
# Shift the first two arguments off
shift 2
# Now $@ contains all remaining arguments

python main.py \
    --batch_size 1 \
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
    --num_select 100 \
    --dataset_file polaris \
    --polaris_path $polaris_path \
    --square_resize_div_64 \
    --use_ema \
    --eval --resume $checkpoint \
    --output_dir output/$model_name \
    export_model "$@"  # Pass all remaining arguments to the python script