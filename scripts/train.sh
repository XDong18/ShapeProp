export CUDA_VISIBLE_DEVICES=5
python shapeprop/tools/train_net.py \
    --config-file configs/bdd_yanzhao_r50.yml \
    --local_rank 1 \
    --num_gpu 1