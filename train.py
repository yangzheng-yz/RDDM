import os
import sys

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)

# init
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
sys.stdout.flush()
set_seed(10)
debug = False

if debug:
    save_and_sample_every = 2
    sampling_timesteps = 2
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 25
else:
    save_and_sample_every = 10
    if len(sys.argv) > 1:
        sampling_timesteps = int(sys.argv[1])
    else:
        sampling_timesteps = 2
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 400000

original_ddim_ddpm = False
if original_ddim_ddpm:
    condition = False
    input_condition = False
    input_condition_mask = False
else:
    condition = True # False
    input_condition = False
    input_condition_mask = False

if condition:
    # Image restoration  
    if input_condition:
        folder = ["xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_mask_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_test.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_test.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_mask_test.flist"]
    else:
        # folder = ["xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_train.flist",
        #           "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_train.flist",
        #           "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_test.flist",
        #           "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_test.flist"]
        folder = ["/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/train/without_art",
                  "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/val/without_art"]
    train_batch_size = 1
    num_samples = 1
    sum_scale = 0.01
    image_size = 256
else:
    # Image Generation 
    folder = 'xxx/CelebA/img_align_celeba'
    train_batch_size = 128
    num_samples = 64
    sum_scale = 1
    image_size = 64

num_unet = 2 # 2
objective = 'pred_res_noise' # 'pred_res_noise'
test_res_or_noise = "res_noise" # "noise"
if original_ddim_ddpm:
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,           # number of steps
        sampling_timesteps=sampling_timesteps_original_ddim_ddpm,
        loss_type='l1',            # L1 or L2
    )
else:
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=num_unet,
        condition=condition,
        input_condition=input_condition,
        objective=objective,
        test_res_or_noise = test_res_or_noise,
        channels=1
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=image_size, # TODO: (512, 640)?
        timesteps=1000,           # number of steps
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        loss_type='l1',            # L1 or L2
        condition=condition,
        sum_scale=sum_scale,
        input_condition=input_condition,
        input_condition_mask=input_condition_mask,
        test_res_or_noise = test_res_or_noise
    )
artifact_source_path="/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/For_argumentation/et_2-500ms-250_550-1000ms-10_1500-5000ms-8_1800LP_contrast.raw"
trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=8e-5,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="L",
    results_folder='./results',
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=False,
    generation=False,
    num_unet=num_unet,
    artifact_property_trian={'random': False, 'fixed_exposure_time': 100, 'artifact_source_path': artifact_source_path},
    artifact_property_val={'random': False, 'fixed_exposure_time': 100, 'artifact_source_path': artifact_source_path}
)

# train
trainer.train()

# test
if not trainer.accelerator.is_local_main_process:
    pass
else:
    trainer.load(trainer.train_num_steps//save_and_sample_every)
    trainer.set_results_folder(
        '/mnt/samsung/zheng_data/training_log/RDDM_deblurring/test_timestep_'+str(sampling_timesteps))
    trainer.test(last=True)

# trainer.set_results_folder('./results/test_sample')
# trainer.test(sample=True)
