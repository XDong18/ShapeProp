# Classwise semi-supervision (VOC categories)
# Train baseline Mask R-CNN
export CUDA_VISIBLE_DEVICES=1,2,3
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --master_port=3000 \
    shapeprop/tools/train_net.py \
    --config-file configs/bdd100k_mask_rcnn_r50_fpn_1x.yml \
    --local_rank 0 \
    --num_gpu 3
# Evaluate
