from easydict import EasyDict as edict

config = edict()

############################################### [START] copied from /home/wxs/ContextAware-PoseFormer/ContextPose/mvn/utils/cfg.py ###############################################
# model definition
config.model = edict()
config.model.image_shape = [192, 256]
config.model.init_weights = True
config.model.checkpoint = None

config.model.backbone = edict()
config.model.backbone.type = 'hrnet_32'
config.model.backbone.num_final_layer_channel = 17
config.model.backbone.num_joints = 17
config.model.backbone.num_layers = 152
config.model.backbone.init_weights = True
config.model.backbone.fix_weights = False
config.model.backbone.checkpoint = "../ContextAware-PoseFormer/ContextPose/data/pretrained/coco/pose_hrnet_w32_256x192.pth"

# pose_hrnet related params
# config.model.backbone = edict()
config.model.backbone.NUM_JOINTS = 17
config.model.backbone.PRETRAINED_LAYERS = ['*']
config.model.backbone.STEM_INPLANES = 64
config.model.backbone.FINAL_CONV_KERNEL = 1

config.model.backbone.STAGE2 = edict()
config.model.backbone.STAGE2.NUM_MODULES = 1
config.model.backbone.STAGE2.NUM_BRANCHES = 2
config.model.backbone.STAGE2.NUM_BLOCKS = [4, 4]
config.model.backbone.STAGE2.NUM_CHANNELS = [32, 64]
# config.model.backbone.STAGE2.NUM_CHANNELS = [48, 96]
config.model.backbone.STAGE2.BLOCK = 'BASIC'
config.model.backbone.STAGE2.FUSE_METHOD = 'SUM'

config.model.backbone.STAGE3 = edict()
# config.model.backbone.STAGE3.NUM_MODULES = 1
config.model.backbone.STAGE3.NUM_MODULES = 4
config.model.backbone.STAGE3.NUM_BRANCHES = 3
config.model.backbone.STAGE3.NUM_BLOCKS = [4, 4, 4]
config.model.backbone.STAGE3.NUM_CHANNELS = [32, 64, 128]
# config.model.backbone.STAGE3.NUM_CHANNELS = [48, 96, 192]
config.model.backbone.STAGE3.BLOCK = 'BASIC'
config.model.backbone.STAGE3.FUSE_METHOD = 'SUM'

config.model.backbone.STAGE4 = edict()
# config.model.backbone.STAGE4.NUM_MODULES = 1
config.model.backbone.STAGE4.NUM_MODULES = 3
config.model.backbone.STAGE4.NUM_BRANCHES = 4
config.model.backbone.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
config.model.backbone.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
# config.model.backbone.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
config.model.backbone.STAGE4.BLOCK = 'BASIC'
config.model.backbone.STAGE4.FUSE_METHOD = 'SUM'

# pose_resnet related params
config.model.backbone.NUM_LAYERS = 50
config.model.backbone.DECONV_WITH_BIAS = False
config.model.backbone.NUM_DECONV_LAYERS = 3
config.model.backbone.NUM_DECONV_FILTERS = [256, 256, 256]
config.model.backbone.NUM_DECONV_KERNELS = [4, 4, 4]
config.model.backbone.FINAL_CONV_KERNEL = 1
config.model.backbone.PRETRAINED_LAYERS = ['*']
############################################### [END] copied from /home/wxs/ContextAware-PoseFormer/ContextPose/mvn/utils/cfg.py ###############################################

config.model.hybrid = edict()
config.model.hybrid.hrnet_output_level = 3  # int or list. 0,1,2,3 分别对应输出 [B,32,H/4,W/4], [B,64,H/8,W/8], [B,128,H/16,W/16], [B,256,H/32,W/32] 的特征
config.model.hybrid.vision_guidance_ratio = 0
config.model.hybrid.vision_guidance_where = 'enc'
config.model.hybrid.vision_guidance_fuse = 'cat'
config.model.hybrid.vision_guidance_extraLoss = None    # infonce

config.model.hybrid.vision_guidance_extraLossConfig = {
    'infonce': {
        'temperature': 0.07,
        'loss_weight': 0.01,
    }
}