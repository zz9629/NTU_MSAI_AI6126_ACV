
==================================

Matriculation Number: G211941J
CodaLab username: ZENG_ZHENG

Best mIOU Score: 75.3807243681

==================================

More informations are as follow: 

----------------- Description of the files -------------
.
├── report.pdf	       ## A short report of this project.
├── Readme.txt         ## This descriptive document.
├── Screenshot.png     ## Screenshot on Codalab of score achieved, the best score is 75.3807243681.
├── checkpoint.pth     ## The model checkpoint (weights) of my submitted model.
├── code               ## All necessary codes I used in this project.
│   ├── config.py                   ## The config of my model
│   ├── generateClassWeights.py     ## A tool script to generate class weight for Weighted Cross-Entropy loss function.
│   └── main.py                     ## The script to generate predicted masks using exsiting checkpoint and config file.
└── results.zip        ## The best results (predicted masks) from my model on the 1000 test images. 

----------------- Third-party libraries-----------------

Pytorch==1.10.0
mmsegmentation==0.22.1 
mmcv-full==1.4.7
opencv-python

----------------- How to test my solution ---------------

1. Make sure you have installed libraries above.
2. Unzip code.zip and go into it, run main.py script. 

Description of arguments of main.py:

- '--in': The folder path of test images.
- '--out': The zip file name of predicted masks.

For example:

```shell
unzip code
cd code
python main.py --in data_folder --out results.zip
```

The final output results.zip will be in current path.