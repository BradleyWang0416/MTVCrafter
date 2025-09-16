# mode=debug
mode=train

EXP_NAME=joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb4096x2048_mpjpe/hrFix_lvl3_ratio0
CONFIG=vqvae_experiment_configs/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb4096x2048_mpjpe/hrFix_lvl3_ratio0/config.yaml
LOG=vqvae_experiment_configs/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb4096x2048_mpjpe/hrFix_lvl3_ratio0/train.log
RESUME_PATH=''

BATCH_SIZE=32

DATA_MODE=joint3d

NUM_CODE=4096   # 8192
CODE_DIM=2048   # 3072
NUM_FRAME=16
SAMPLE_STRIDE=1
DATA_STRIDE=16

LOSS_TYPE=mpjpe     # l1, mpjpe

VQVAE_TYPE=hybrid  # base, hybrid
HRNET_OUTPUT_LEVEL=3    # int or list. 0,1,2,3 分别对应输出 [B,32,H/4,W/4], [B,64,H/8,W/8], [B,128,H/16,W/16], [B,256,H/32,W/32] 的特征
VISION_GUIDANCE_RATIO=0

FIX_WEIGHTS=True


if [ "$FIX_WEIGHTS" = "True" ]; then
    FIX_WEIGHTS_ARG="--fix_weights"
else
    FIX_WEIGHTS_ARG=""
fi

if [ "$mode" = "debug" ]; then
    CUDA_VISIBLE_DEVICES=5 \
        python \
        -m debugpy --listen 5680 --wait-for-client \
        train_vqvae_new.py \
        --batch_size ${BATCH_SIZE} \
        --config ${CONFIG} \
        --data_mode ${DATA_MODE} \
        --num_frames ${NUM_FRAME} \
        --sample_stride ${SAMPLE_STRIDE} \
        --data_stride ${DATA_STRIDE} \
        --project_dir vqvae_experiment/tmp \
        --not_find_unused_parameters \
        --nb_code ${NUM_CODE} \
        --codebook_dim ${CODE_DIM} \
        --loss_type ${LOSS_TYPE} \
        --vqvae_type ${VQVAE_TYPE} \
        --hrnet_output_level ${HRNET_OUTPUT_LEVEL} \
        --vision_guidance_ratio ${VISION_GUIDANCE_RATIO} \
        $FIX_WEIGHTS_ARG \
        --resume_pth "${RESUME_PATH}"
else
    CUDA_VISIBLE_DEVICES=1 \
        nohup \
        python -u train_vqvae_new.py \
        --batch_size ${BATCH_SIZE} \
        --config ${CONFIG} \
        --data_mode ${DATA_MODE} \
        --num_frames ${NUM_FRAME} \
        --sample_stride ${SAMPLE_STRIDE} \
        --data_stride ${DATA_STRIDE} \
        --project_dir vqvae_experiment/${EXP_NAME} \
        --not_find_unused_parameters \
        --nb_code ${NUM_CODE} \
        --codebook_dim ${CODE_DIM} \
        --loss_type ${LOSS_TYPE} \
        --vqvae_type ${VQVAE_TYPE} \
        --hrnet_output_level ${HRNET_OUTPUT_LEVEL} \
        --vision_guidance_ratio ${VISION_GUIDANCE_RATIO} \
        $FIX_WEIGHTS_ARG \
        --resume_pth "${RESUME_PATH}" \
        > ${LOG} &
fi