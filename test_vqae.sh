NUM_CODE=4096   # 8192
CODE_DIM=2048   # 3072
NUM_FRAME=64
SAMPLE_STRIDE=1
DATA_MODE=joint3d
CKPT_PATH=vqvae_experiment/h36m_j3d_f64s1_cb4096x2048/models/checkpoint_epoch_665_step_500000
LOSS_TYPE=mpjpe     # l1, mpjpe


    # -m debugpy --listen 5680 --wait-for-client \
CUDA_VISIBLE_DEVICES=6 \
    python \
    test_vqvae.py \
    --resume_pth ${CKPT_PATH} \
    --load_data_file "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl" \
    --num_frames ${NUM_FRAME} \
    --sample_stride ${SAMPLE_STRIDE} \
    --nb_code ${NUM_CODE} \
    --codebook_dim ${CODE_DIM} \
    --data_mode ${DATA_MODE} \
    --loss_type ${LOSS_TYPE} \
    --num_vis_samples 0
