mode=debug
# mode=train

NUM_CODE=4096   # 8192
CODE_DIM=2048   # 3072
NUM_FRAME=64
SAMPLE_STRIDE=1

LOSS_TYPE=l1     # l1, mpjpe


EXP_NAME=h36m_j3d_f64s1_cb4096x2048
DATA_MODE=joint3d
LOG=h36m_j3d_f64s1_cb4096x2048.log


# D1="/data2/wxs/DATASETS/PW3D_ByBradley/all_data.pkl"
D2="/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl"
# D3="/data2/wxs/DATASETS/AMASS_ByBradley/"
# DATA_PATH=${D1},${D2},${D3}
DATA_PATH=${D2}



if [ "$mode" = "debug" ]; then
    CUDA_VISIBLE_DEVICES=5 \
        python \
        -m debugpy --listen 5680 --wait-for-client \
        train_vqvae.py \
        --data_mode ${DATA_MODE} \
        --load_data_file ${DATA_PATH} \
        --num_frames ${NUM_FRAME} \
        --sample_stride ${SAMPLE_STRIDE} \
        --project_dir vqvae_experiment/tmp \
        --not_find_unused_parameters \
        --nb_code ${NUM_CODE} \
        --codebook_dim ${CODE_DIM} \
        --loss_type ${LOSS_TYPE}
else
    CUDA_VISIBLE_DEVICES=5 \
        nohup \
        python -u train_vqvae.py \
        --data_mode ${DATA_MODE} \
        --load_data_file ${DATA_PATH} \
        --num_frames ${NUM_FRAME} \
        --sample_stride ${SAMPLE_STRIDE} \
        --project_dir vqvae_experiment/${EXP_NAME} \
        --not_find_unused_parameters \
        --nb_code ${NUM_CODE} \
        --codebook_dim ${CODE_DIM} \
        --loss_type ${LOSS_TYPE} \
        > ${LOG} &
fi