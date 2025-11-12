# mode=debug
# mode=train
mode=test

EXP_NAME="joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb8192x2048_mpjpe_Tdown1-2/hrFix_lvl3_ratio0.5"




# CONFIG="vqvae_experiment_configs/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb8192x2048_mpjpe_Tdown1-2/hrFix_lvl3_ratio0.5/config.yaml"
CONFIG="vqvae_experiment_configs/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb8192x2048_mpjpe_Tdown1-2/hrFix_lvl3_ratio0.5/config_3dpw.yaml"



LOG="vqvae_experiment_configs/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb8192x2048_mpjpe_Tdown1-2/hrFix_lvl3_ratio0.5/train.log"

if [ "$mode" = "test" ]; then
    RESUME_PATH="vqvae_experiment/joint_and_image/joint3d_image_affined_192x256/f16s1d16_cb8192x2048_mpjpe_Tdown1-2/hrFix_lvl3_ratio0.5/models/checkpoint_epoch_395_step_300000"
    LOSS_TYPE=mpjpe_millimeter     # l1, mpjpe, mpjpe_millimeter
    BATCH_SIZE=32
else
    RESUME_PATH=""
    LOSS_TYPE=mpjpe     # l1, mpjpe
    BATCH_SIZE=64
fi


DATA_MODE=joint3d

NUM_CODE=8192   # 8192
CODE_DIM=2048   # 3072
NUM_FRAME=16
SAMPLE_STRIDE=1
DATA_STRIDE=16


VQVAE_TYPE=hybrid  # base, hybrid
HRNET_OUTPUT_LEVEL=3    # int or list. 0,1,2,3 分别对应输出 [B,32,H/4,W/4], [B,64,H/8,W/8], [B,128,H/16,W/16], [B,256,H/32,W/32] 的特征
VISION_GUIDANCE_RATIO=0.5

FIX_WEIGHTS=True


if [ "$FIX_WEIGHTS" = "True" ]; then
    FIX_WEIGHTS_ARG="--fix_weights"
else
    FIX_WEIGHTS_ARG=""
fi

if [ "$mode" = "debug" ]; then
    CUDA_VISIBLE_DEVICES=7 \
        python \
        -m debugpy --listen 5678 --wait-for-client \
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
        --downsample_time "[1,2]" \
        --frame_upsample_rate "[2.0,1.0]" \
        $FIX_WEIGHTS_ARG \
        --resume_pth "${RESUME_PATH}"
elif [ "$mode" = "test" ]; then
        # -m debugpy --listen 5680 --wait-for-client \
        # accelerate launch --num_processes 5 \
        # -m debugpy --listen 5678 --wait-for-client \
    CUDA_VISIBLE_DEVICES=3 \
        python \
        test_vqvae_new.py \
        --config ${CONFIG} \
        --resume_pth "${RESUME_PATH}" \
        --batch_size ${BATCH_SIZE} \
        --nb_code ${NUM_CODE} \
        --codebook_dim ${CODE_DIM} \
        --loss_type ${LOSS_TYPE} \
        --vqvae_type ${VQVAE_TYPE} \
        --num_frames ${NUM_FRAME} \
        --sample_stride ${SAMPLE_STRIDE} \
        --data_stride ${DATA_STRIDE} \
        --data_mode ${DATA_MODE} \
        --hrnet_output_level ${HRNET_OUTPUT_LEVEL} \
        --vision_guidance_ratio ${VISION_GUIDANCE_RATIO} \
        --downsample_time "[1,2]" \
        --frame_upsample_rate "[2.0,1.0]"
else
        # python -u train_vqvae_new.py \
    CUDA_VISIBLE_DEVICES=6,7 \
        nohup \
        accelerate launch --num_processes 2 --main_process_port 29233 \
        train_vqvae_new.py \
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
        --downsample_time "[1,2]" \
        --frame_upsample_rate "[2.0,1.0]" \
        $FIX_WEIGHTS_ARG \
        --resume_pth "${RESUME_PATH}" \
        > ${LOG} &
fi