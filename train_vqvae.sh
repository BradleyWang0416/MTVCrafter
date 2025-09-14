# EXP_NAME="all_datasets"
# DATA_MODE="joint3d"
EXP_NAME="all_datasets_j3d_f64s2"
DATA_MODE="joint3d"

D1="/data2/wxs/DATASETS/PW3D_ByBradley/all_data.pkl"
D2="/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl"
D3="/data2/wxs/DATASETS/AMASS_ByBradley/"
DATA_PATH=${D1},${D2},${D3}

CUDA_VISIBLE_DEVICES=7 \
nohup \
    python train_vqvae.py \
    --data_mode ${DATA_MODE} \
    --load_data_file ${DATA_PATH} \
    --num_frames 64 \
    --sample_stride 2 \
    --project_dir vqvae_experiment/${EXP_NAME} \
    --not_find_unused_parameters \
    > all_datasets_j3d_f64s2.log &