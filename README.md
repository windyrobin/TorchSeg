# TorchSeg
This project aims at providing a fast, modular reference implementation for semantic segmentation models using PyTorch.

![demo image](demo/cityscapes_demo_img.png)

## Highlights
- **Modular Design:** easily construct a customized semantic segmentation models by combining different components.
- **Distributed Training:** **>60%** faster than the multi-thread parallel method([nn.DataParallel](https://pytorch.org/docs/stable/nn.html#dataparallel)), we use the multi-processing parallel method.
- **Multi-GPU training and inference:** support different manners of inference.
- Provides pre-trained models and implement different semantic segmentation models.

## Prerequisites
- PyTorch 1.0
  - `pip3 install torch torchvision`
- Easydict
  - `pip3 install easydict`
- [Apex](https://nvidia.github.io/apex/index.html)
- Ninja
  - `sudo apt-get install ninja-build`
- tqdm
  - `pip3 install tqdm`

## Model Zoo
### Supported Model
- FCN
- [DFN](https://arxiv.org/abs/1804.09337) 
- [BiSeNet](https://arxiv.org/abs/1808.00897)
- PSPNet

### Performance and Benchmarks
SS:Single Scale MSF:Multi-scale + Flip

### PASCAL VOC 2012
 Methods | Backbone | TrainSet | EvalSet | Mean IoU(SS) | Mean IoU(MSF) | Model 
:--:|:--:|:--:|:--:|:--:|:--:|:--:
 FCN-32s     | R101_v1c | *train_aug*  | *val*  | 71.26 | -                 | BaiduYun / GoogleDrive 
 DFN(paper)  | R101_v1c | *train_aug*  | *val*  | 79.67 | 80.6<sup>1</sup>  | BaiduYun / GoogleDrive 
 DFN(ours)   | R101_v1c | *train_aug*  | *val*  | 79.63 | 81.15             | BaiduYun / GoogleDrive 


80.6<sup>1</sup>: this result reported in paper is further finetuned on *train* dataset. 

### Cityscapes
#### Non-real-time Methods
 Methods | Backbone |OHEM| TrainSet | EvalSet | Mean IoU(ss) | Mean IoU(msf) | Model 
:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:
 DFN(paper)                 | R101_v1c |✗| *train_fine* | *val*  | 78.5  | 79.3   | BaiduYun / GoogleDrive 
 DFN(ours)                  | R101_v1c |✓| *train_fine* | *val*  | 79.49 | 80.32  | BaiduYun / GoogleDrive 
 BiSeNet(paper)             | R101_v1c |✓| *train_fine* | *val*  |  -    | 80.3   | BaiduYun / GoogleDrive 
 BiSeNet(ours)              | R101_v1c |✓| *train_fine* | *val*  | 79.56 | 80.29  | BaiduYun / GoogleDrive 
 BiSeNet(paper)             | R18      |✓| *train_fine* | *val*  | 76.21 | 78.57  | BaiduYun / GoogleDrive 
 BiSeNet(ours)              | R18      |✓| *train_fine* | *val*  | 76.33 | 78.46  | BaiduYun / GoogleDrive 
 BiSeNet(paper)             | X39      |✓| *train_fine* | *val*  | 70.1  | 72     | BaiduYun / GoogleDrive 
 BiSeNet(ours)<sup>1</sup>  | X39      |✓| *train_fine* | *val*  | 69.1  | 72.2   | BaiduYun / GoogleDrive 
 
 BiSeNet(ours)<sup>1</sup>: because we didn't pre-train the Xception39 model on ImageNet in PyTorch, we train this experiment from scratch. We will release the pre-trained Xception39 model in PyTorch and the corresponding experiment.
 
 #### Real-time Methods
  Methods | Backbone |OHEM| TrainSet | EvalSet | Mean IoU | Model 
:--:|:--:|:--:|:--:|:--:|:--:|:--:
 BiSeNet(paper)             | R18      |✓| *train_fine* | *val*  | 74.8 | BaiduYun / GoogleDrive 
 BiSeNet(ours)              | R18      |✓| *train_fine* | *val*  | 74.6 | BaiduYun / GoogleDrive 
 BiSeNet(paper)             | X39      |✓| *train_fine* | *val*  | 69   | BaiduYun / GoogleDrive 
 BiSeNet(ours)<sup>1</sup>  | X39      |✓| *train_fine* | *val*  | 68.5 | BaiduYun / GoogleDrive 
 
### ADE
  Methods | Backbone | TrainSet | EvalSet | Mean IoU | Accuracy|  Model 
:--:|:--:|:--:|:--:|:--:|:--:|:--:
 PSPNet(paper) | R50_v1c | *train* | *val*  | 41.68(ss) | 80.04(ss) | BaiduYun / GoogleDrive 
 PSPNet(ours)  | R50_v1c | *train* | *val*  | 41.61(ss) | 80.19(ss) | BaiduYun / GoogleDrive 

### To Do
- [ ] release all trained models
- [ ] offer comprehensive documents 
- [ ] support more semantic segmentation models
  - [ ] Deeplab v3 / Deeplab v3+
  - [ ] DenseASPP
  - [ ] PSANet
  - [ ] EncNet
  - [ ] OCNet


## Training
1. create the config file of dataset:`train.txt`, `val.txt`, `test.txt`   
    file structure：(split with `tab`)
    ```txt
    path-of-the-image   path-of-the-groundtruth
    ```
2. modify the `config.py` according to your requirements
3. train a network:

### Distributed Training
We use the official `torch.distributed.launch` in order to launch multi-gpu training. This utility function from PyTorch spawns as many Python processes as the number of GPUs we want to use, and each Python process will only use a single GPU.

For each experiment, you can just run this script:
```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
```

### Non-distributed Training
The above performance are all conducted based on the non-distributed training.
For each experiment, you can just run this script:
```bash
python train.py -d 0-7
```
the argument of `d` means the GPU you want to use.

## Inference
In the evaluator, we have implemented the multi-gpu inference base on the multi-process. In the inference phase, the function will spawns as many Python processes as the number of GPUs we want to use, and each Python process will handle a subset of the whole evaluation dataset on a single GPU.
1. evaluate a trained network on the validation set:
    ```bash
    python3 eval.py
    ```
2. input arguments:
    ```bash
    usage: -e epoch_idx -d device_idx [--verbose ] 
    [--show_image] [--save_path Pred_Save_Path]
    ```
3. sample here
add pytrt5 support:
```
 c++ -O3 -Wall -shared -std=c++11 -fPIC -I /mnt/pybind11/include/ -I/usr/include/python3.5 example.cpp -lnvonnxparser_runtime  -o example`python3-config --extension-suffix`
```

```
root@gpu-labeling:/mnt/xm-bisenet/model/bisenet/cityscapes.bisenet.R18.speed# python3 infer.py --tensorrt                                                                                                   
09 10:28:08 using devices 1                                                                                                                                                                                 
Reading engine from file ./bisenet-half.trt                                                                                                                                                                 
<tensorrt.tensorrt.IPluginFactory object at 0x7f0ea7275538>                                                                                                                                                 
[TensorRT] INFO: Glob Size is 64269120 bytes.                                                                                                                                                               
[TensorRT] INFO: Added linear block of size 75497472                                                                                                                                                        
[TensorRT] INFO: Added linear block of size 44826624                                                                                                                                                        
[TensorRT] INFO: Added linear block of size 9437184                                                                                                                                                         
[TensorRT] INFO: Added linear block of size 9437184                                                                                                                                                         
[TensorRT] INFO: Added linear block of size 4718592                                                                                                                                                         
[TensorRT] INFO: Added linear block of size 1179648                                                                                                                                                         
[TensorRT] INFO: Added linear block of size 512                                                                                                                                                             
[TensorRT] WARNING: TensorRT was compiled against cuDNN 7.5.0 but is linked against cuDNN 7.5.1. This mismatch may potentially cause undefined behavior.                                                    
[TensorRT] INFO: Deserialize required 28194 microseconds.                                                                                                                                                   
[TensorRT] WARNING: TensorRT was compiled against cuDNN 7.5.0 but is linked against cuDNN 7.5.1. This mismatch may potentially cause undefined behavior.                                                    
binding: 0                                                                                                                                                                                                  
shape: (3, 768, 1536)                                                                                                                                                                                       
nptype: DataType.FLOAT                                                                                                                                                                                      
input?: True                                                                                                                                                                                                
binding: 340                                                                                                                                                                                                
shape: (768, 1536)                                                                                                                                                                                          
nptype: DataType.INT32                                                                                                                                                                                      
input?: False                                                                                                                                                                                               
input image:                                                                                                                                                                                                
input shape: (3, 768, 1536)                                                                                                                                                                                 
max,min,mean,std: 0.594 -0.485 0.31385097 0.20309699                                                                                                                                                        
infer cost: 13              
```

## Disclaimer
This project is under active development. So things that are currently working might break in a future release. However, feel free to open issue if you get stuck anywhere.

## Citation
The following are BibTeX references. The BibTeX entry requires the url LaTeX package.

Please consider citing this project in your publications if it helps your research. 
```
@misc{torchseg2019,
  author =       {Yu, Changqian},
  title =        {TorchSeg},
  howpublished = {\url{https://github.com/ycszen/TorchSeg}},
  year =         {2019}
}
```

Please consider citing the [DFN](https://arxiv.org/abs/1804.09337) in your publications if it helps your research. 

```
@article{yu2018dfn,
  title={Learning a Discriminative Feature Network for Semantic Segmentation},
  author={Yu, Changqian and Wang, Jingbo and Peng, Chao and Gao, Changxin and Yu, Gang and Sang, Nong},
  journal={arXiv preprint arXiv:1804.09337},
  year={2018}
}
```

Please consider citing the [BiSeNet](https://arxiv.org/abs/1808.00897) in your publications if it helps your research. 

```
@inproceedings{yu2018bisenet,
  title={Bisenet: Bilateral segmentation network for real-time semantic segmentation},
  author={Yu, Changqian and Wang, Jingbo and Peng, Chao and Gao, Changxin and Yu, Gang and Sang, Nong},
  booktitle={European Conference on Computer Vision},
  pages={334--349},
  year={2018},
  organization={Springer}
}
```

## Why this name, Furnace?
Furnace means the **Alchemical Furnace**. We all are the **Alchemist**, so I hope everyone can have a good alchemical furnace to practice the **Alchemy**. Hope you can be a excellent alchemist. 

