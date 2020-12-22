export CUDA_VISIBLE_DEVICES=1
python shapeprop/tools/train_net.py \
    --config-file configs/bdd100k_mask_rcnn_r50_fpn_1x.yml \
    --local_rank 1 \
    --num_gpu 1