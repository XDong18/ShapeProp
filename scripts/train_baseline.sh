# Classwise semi-supervision (VOC categories)
# Train baseline Mask R-CNN
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=3000 \
    shapeprop/tools/train_net.py \
    --config-file configs/bdd100k_mask_rcnn_r50_fpn_1x.yml \
    --local_rank 0 \
    --num_gpu 4
# Evaluate
