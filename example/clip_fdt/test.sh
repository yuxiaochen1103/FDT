MODEL_FOLD='/research/cbim/medical/yc984/Code_clip/train_log/cc3m/clip_fdt'
DATA_FOLD='/research/cbim/archive/yc984/img_dataset'


ITER='86560'

bash run.sh example/clip_fdt/ZS_CLS_evaluator.py \
        --config ${MODEL_FOLD}/config.json \
        --ckpt_path ${MODEL_FOLD}/checkpoints/ckpt_${ITER}.pth.tar \
        --output_path ${MODEL_FOLD}/results \
        --data_fold ${DATA_FOLD}

bash run.sh example/clip_fdt/coco_evaluator.py \
        --config ${MODEL_FOLD}/config.json \
        --ckpt_path ${MODEL_FOLD}/checkpoints/ckpt_${ITER}.pth.tar \
        --output_path ${MODEL_FOLD}/results \
        --data_fold ${DATA_FOLD}
