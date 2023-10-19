import os
import sys

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)

# init
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
sys.stdout.flush()
set_seed(10)
debug = False

if debug:
    save_and_sample_every = 2
    sampling_timesteps = 2
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 25
else:
    save_and_sample_every = 1000
    # if len(sys.argv) > 1:
    #     sampling_timesteps = int(sys.argv[1])
    # else:
    sampling_timesteps = 5
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
                  "/mnt/samsung/zheng_data/datasets/NIRI_to_NIRII/train/without_art"]
    train_batch_size = 9
    num_samples = 9
    sum_scale = 0.01
    image_size = 64
else:
    # Image Generation 
    folder = 'xxx/CelebA/img_align_celeba'
    train_batch_size = 128
    num_samples = 64
    sum_scale = 1
    image_size = 64

num_unet = 1 # 2
objective = 'pred_res' # 'pred_res_noise'
test_res_or_noise = "res" # "noise"
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
    results_folder='/mnt/samsung/zheng_data/training_log/RDDM_deblurring_resonly_64',
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=False,
    generation=False,
    num_unet=num_unet,
    artifact_property_trian={'random': False, 'fixed_exposure_time': 100, 'artifact_source_path': artifact_source_path},
    artifact_property_val={'random': False, 'fixed_exposure_time': 100, 'artifact_source_path': artifact_source_path}
)

# load latest model
files = os.listdir("/mnt/samsung/zheng_data/training_log/RDDM_deblurring_resonly_64")
models_no = [int(i.split('.')[0].split('-')[-1]) for i in files if 'model-' in i]
models_no.sort()
if len(models_no) == 0:
    print(f"No pretrained model")
else:
    if trainer.accelerator.is_main_process:
        trainer.load(models_no[-1])
        # print(f"Load latest model of {models_no[-1]}")
    
        # if trainer.step != 0 and trainer.step % (trainer.save_and_sample_every*10) == 0:
        #     results_folder = trainer.results_folder
        #     try:
        #         gen_img = f'/mnt/samsung/zheng_data/training_log/RDDM_deblurring_resonly_64/test_timestep_{trainer.model.sampling_timesteps}_' + \
        #             str(models_no[-1])+"_pt"
        #     except:
        #         gen_img = f'/mnt/samsung/zheng_data/training_log/RDDM_deblurring_resonly_64/test_timestep_{trainer.model.module.sampling_timesteps}_' + \
        #             str(models_no[-1])+"_pt"    
        #     trainer.set_results_folder(gen_img)
        #     trainer.test(last=True, FID=False, sample=True)
        #     trainer.set_results_folder(results_folder)

# train
# trainer.train()

# test
if not trainer.accelerator.is_local_main_process:
    pass
else:
    trainer.load(110)
    trainer.set_results_folder(
        '/mnt/samsung/zheng_data/training_log/RDDM_deblurring_resonly_newart/test_timestep_'+str(sampling_timesteps))
    trainer.test(last=True)

# trainer.set_results_folder('./results/test_sample')
# trainer.test(sample=True)
