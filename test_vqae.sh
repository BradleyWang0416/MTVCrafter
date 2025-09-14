CUDA_VISIBLE_DEVICES=6 \
    python -m debugpy --listen 5680 --wait-for-client test_vqvae.py \
    --resume_pth vqvae_experiment/amass_j3d/models/checkpoint_epoch_129_step_220000 \
    --load_data_file "/data2/wxs/DATASETS/Human3.6M_for_MotionBERT/h36m_sh_conf_cam_source_final.pkl" \
    --data_mode joint3d \
    --num_vis_samples 0
