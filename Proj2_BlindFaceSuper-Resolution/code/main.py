# Check Pytorch installation
import torch, torchvision

print(torch.__version__, torch.cuda.is_available())

# Check MMEditing installation
import mmedit

print(mmedit.__version__)

from mmcv import Config
# Load the original config

import os.path as osp

from mmedit.datasets import build_dataset
from mmedit.models import build_model
from mmedit.apis import train_model
from mmcv.runner import init_dist

import mmcv
import os
from mmcv.runner import set_random_seed

from mmedit.apis import single_gpu_test
from mmedit.datasets import build_dataloader
from mmcv.parallel import MMDataParallel

import cv2
import matplotlib.pyplot as plt
import mmcv
import torch
import torchvision

from mmedit.models import build_model
from mmcv.runner import load_checkpoint


def make_folder(path, version):
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))


def saveConfig(cfg):
    make_folder(cfg.work_dir, '')
    with open(osp.join(cfg.work_dir, 'cofig.py'), 'w') as f:
        f.write(cfg.pretty_text)
    f.close()


def train(config_file):
    # cfg = Config.fromfile('configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py')
    cfg = Config.fromfile(config_file)
    print(f'Config:\n{cfg.pretty_text}')  # Show the config

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpus = 1

    # Initialize distributed training (only need to initialize once), comment it if
    # have already run this part
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '50297'
    # init_dist('pytorch', **cfg.dist_params)

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the SRCNN model
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    saveConfig(cfg)

    # Meta information
    meta = dict()
    if cfg.get('exp_name', None) is None:
        cfg['exp_name'] = osp.splitext(osp.basename(cfg.work_dir))[0]
    meta['exp_name'] = cfg.exp_name
    meta['mmedit Version'] = mmedit.__version__
    meta['seed'] = 0

    # Train the model
    train_model(model, datasets, cfg, distributed=True, validate=True, meta=meta)


def inference(checkpoint_file, config_file, cfg):
    cfg = Config.fromfile(config_file)
    print(f'Config:\n{cfg.pretty_text}')

    model = build_model(cfg.model).cuda()
    load_checkpoint(model, checkpoint_file, map_location='cuda')

    # Build a test dataloader and model
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        persistent_workers=False)
    model = MMDataParallel(model, device_ids=[0])

    # Perform the test with single gpu. Saving result images by setting save_image and save_path arguments. The two
    # arguments will be passed to model.forword_test() where images are saved. See
    # https://github.com/open-mmlab/mmediting/blob/8b5c0c5f49e60fd6ab0503031b62dee7832faf72/mmedit/models/mattors
    # /indexnet.py#L72.
    outputs = single_gpu_test(model, data_loader, save_image=True,
                              # save_path='./tutorial_exps/srcnn/results')
                              save_path=os.path.join(cfg.work_dir, 'results'))

    # Pop out some unnecessary arguments
    eval_config = cfg.evaluation
    eval_config.pop('interval')
    eval_config.pop('save_image', False)
    eval_config.pop('save_path', None)



if __name__ == '__main__':
    config_file = 'srresnet_ffhq_400k.py'
    # train(config_file)

    checkpoint_file = '../checkpoint.pth'
    inference(checkpoint_file, config_file)

    cfg = mmcv.Config.fromfile(config_file)
    predicts_folder = os.path.join(cfg.work_dir, 'results')
    !zip - r - j 'results.zip' $predicts_folder

