# Check Pytorch installation # 1.11.0 True
import cv2
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMSegmentation installation # 0.22.1
import mmseg
print(mmseg.__version__)

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import mmcv
from mmcv import Config
import random
from argparse import ArgumentParser
from zipfile import ZipFile
from os.path import basename


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_folder(path, version):
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))


def saveConfig(cfg):
    make_folder(cfg.work_dir, '')
    with open(osp.join(cfg.work_dir, cfg.expt_name + '_cofig.py'), 'w') as f:
        f.write(cfg.pretty_text)
    f.close()


def zipFiles(predicts_folder, results_zip_name):
    # create a ZipFile object
    with ZipFile(results_zip_name, 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(predicts_folder):
            for filename in filenames:
                # create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath, basename(filePath))


seed_torch(0)

data_folder = 'AI6126_dataset_public/test/test_image'
predicts_folder = 'AI6126_dataset_public/test/test_pred'

# define class and plaette for better visualization
classes = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
           'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
palette = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
           [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
           [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]


@DATASETS.register_module()
class FaceParsingDataset(CustomDataset):  # StanfordBackgroundDataset
    CLASSES = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
               'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
    PALETTE = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
               [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
               [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', split=split, seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
        #   ~/AI6126_dataset_public/img_dir/train_image


def generateMasks(model, data_folder, predicts_folder):
    make_folder(osp.abspath(predicts_folder), '')
    i = 1
    for file in mmcv.scandir(data_folder, suffix='.jpg'):
        print(i, ': ', osp.join(data_folder, file))
        filename = osp.join(predicts_folder, osp.splitext(file)[0] + '.png')

        img = mmcv.imread(osp.join(data_folder, file))
        result = inference_segmentor(model, img)
        # print(result[0])
        cv2.imwrite(filename, result[0])
        i += 1


def train(config_file):
    # define class and plaette for better visualization
    # data_root = 'AI6126_dataset_public'
    cfg = Config.fromfile(config_file)
    print(f'Config:\n{cfg.pretty_text}')

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model.CLASSES = datasets[0].CLASSES

    # the number of parameters of model
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)) 

    # Create work_dir and train
    make_folder(osp.abspath(cfg.work_dir), '')
    saveConfig(cfg)
    train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                    meta=dict())

    # test
    model.cfg = cfg
    model.eval()
    generateMasks(model, data_folder, predicts_folder)

    
# build the model from a config file and a checkpoint file
def inference(checkpoint_file, config_file, data_folder, predicts_folder):
    cfg = Config.fromfile(config_file)
    print(f'Config:\n{cfg.pretty_text}')
    model = init_segmentor(cfg, checkpoint_file, device='cuda:0')
    generateMasks(model, data_folder, predicts_folder)



parser = ArgumentParser()
parser.add_argument("--in", dest='data_folder', type=str, default="test_image", help="The folder path of test images.")
# parser.add_argument("--out", dest='predicts_folder', type=str, default="test_predict", help="The folder path for predicted masks.")
parser.add_argument("--out", dest='results_zip_name', type=str, default="results.zip",
                    help="The zip file name of predicted masks.")

if __name__ == '__main__':
    # 1. train and inference
    # train(config_file)

    # 2.inference
    args = parser.parse_args()
    data_folder = args.data_folder
    predicts_folder = 'test_predict'
    results_zip_name = args.results_zip_name
    config_file = 'config.py'
    checkpoint_file = '../checkpoint.pth'

    inference(checkpoint_file, config_file, data_folder, predicts_folder)

    zipFiles(predicts_folder, results_zip_name)
