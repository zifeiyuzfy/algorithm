# 集成YOLOv11+Faster RCNN+SSD+MCNN 的人群检测和计数算法框架



[TOC]

## 一、框架概述

本环境配置文档适用于集成 **YOLOv11**（目标检测）、**Faster RCNN**（高精度检测）、**SSD**（多尺度检测）和 **MCNN**（密度估计）的综合算法框架，支持目标检测、目标跟踪及密集场景人群计数。MCNN 通过多列卷积网络生成密度图，解决密集遮挡场景下的人群统计问题，与其他算法形成场景互补。

## 二、环境配置

### 2.1 安装CUDA和cuDNN（按自己显卡版本适合的版本安装即可，用于算法的训练）

### 2.2 安装Anaconda3和PyCharm

注意Anaconda3镜像源的配置：

.condarc是conda 应用程序的配置文件，在用户家目录（windows：C:\users\username\），用于管理镜像源。如果不存在，则打开conda的，执行一下：

    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

修改.condarc文件内容为：

```bash
channels:
  - https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - defaults
show_channel_urls: true
channel_alias: http://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
offline: false
```

### 2.3 创建python虚拟环境

虚拟环境的作用：虚拟环境可以隔离不同项目的依赖项，避免不同项目之间的干扰，使项目更加稳定。

进入anaconda prompt，输入以下指令：

```bash
conda create -n algorithm python=3.8 -y  
conda activate algorithm 
```

输入命令可以查看虚拟环境否创建成功：

```
conda env list
```

备注：如果需要删除刚刚创建的虚拟环境，可以通过如下命令删除：

```
conda remove -n algorithm --all
```

### 2.4 安装依赖

#### 2.4.1 PyTorch

torch官网：[Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)

选择和CUDA对应的torch版本（前面自己已经安装的版本），然后复制指令，进入pycharm，点击【Teminal】，进入虚拟环境之后，粘贴上面复杂的torch版本进行安装（然后等待安装完成）

#### 2.4.2 算法专属依赖

1. MCNN 依赖

   ```bash
   pip install h5py==2.10.0  # 密度图存储格式  
   pip install scipy==1.7.3  # 积分计算依赖  
   pip install matplotlib==3.5.1  # 密度图可视化  
   ```

2. **YOLOv11/Faster RCNN/SSD 依赖**：

   新建requirements.txt这个文件到项目根目录下，将下面的内容复制进去

   ```
   certifi==2022.12.7
   charset-normalizer==2.1.1
   colorama==0.4.6
   contourpy==1.2.0
   cycler==0.12.1
   filelock==3.9.0
   fonttools==4.50.0
   fsspec==2024.6.1
   huggingface-hub==0.23.4
   idna==3.4
   Jinja2==3.1.2
   kiwisolver==1.4.5
   MarkupSafe==2.1.3
   matplotlib==3.8.3
   mpmath==1.3.0
   networkx==3.2.1
   numpy==1.26.3
   opencv-python==4.9.0.80
   packaging==24.0
   pandas==2.2.1
   pillow==8.2.0
   psutil==5.9.8
   py-cpuinfo==9.0.0
   pyparsing==3.1.2
   python-dateutil==2.9.0.post0
   pytz==2024.1
   PyYAML==6.0.1
   requests==2.28.1
   scipy==1.12.0
   seaborn==0.13.2
   six==1.16.0
   sympy==1.12
   thop==0.1.1.post2209072238
   torch==2.0.0+cu118
   torchaudio==2.0.1+cu118
   torchvision==0.15.1+cu118
   tqdm==4.60.0
   typing_extensions==4.8.0
   tzdata==2024.1
   ultralytics==8.1.34
   urllib3==1.26.13
   ```

   接着在控制台输入：

   ```
   pip install -r requirements.txt
   ```

## 三、数据集准备

| 算法     | 输入格式       | 数据集                                                       |
| -------- | -------------- | ------------------------------------------------------------ |
| MCNN     | RGB 图像 + CSV | `Shanghaitech`                                               |
| 其他算法 | YOLO/VOC 格式  | `基于COCO2017和VisDrone2019的自建数据集（包含两类：0类pedestrian，1类people）` |

Shanghaitech的下载链接：[ShanghaiTech Part A/B_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/datasetdetail/61019)

自建数据集的链接如下：

txt格式（YOLO)：[CocoAndVis_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/datasetdetail/331982)

voc格式（Faster R-CNN/SSD）：[CocoAndVis_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/datasetdetail/331983)

## 四、核心代码文件简介

algorithmService/algorithm里面是四个算法的核心代码，每个算法为一个文件夹。

### 4.1 YOLO

#### 4.1.1 yolov11_train.py

用于训练模型，可以调整训练参数。

#### 4.1.2 test.py

可加载已经训练好的权重（runs/detect/train5/weights/best.pt）进行图片上人群的检测和计数，修改image_paths里的内容即可选择要预测的图片。

#### 4.1.3 PeopleCount.py

用于人群的检测和计数，可以选择图片或者视频模式，可以修改预测时的参数，默认加载已训练的权重，检测时会弹出窗口显示，检测后也可存储检测后的视频。

### 4.2 Faster R-CNN

#### 4.2.1 train.py

用于训练模型，可以调整训练参数。

#### 4.2.2 frcnn.py

用于设置预测时的参数。

#### 4.2.3 predict.py

加载frcnn.py中设置的参数进行人群的检测，支持单张图片的检测，视频检测，fps测试，遍历文件夹中的图片检测。

#### 4.2.4 PeopleCount.py

用于人群的检测和计数，可以选择图片或者视频模式，默认加载frcnn.py中设置的参数，检测时会弹出窗口显示，检测后也可存储检测后的视频。

### 4.3 SSD

#### 4.3.1 train.py

用于训练模型，可以调整训练参数。

#### 4.3.2 ssd.py

用于设置预测时的参数。

#### 4.3.3 predict.py

加载ssd.py中设置的参数进行人群的检测，支持单张图片的检测，视频检测，fps测试，遍历文件夹中的图片检测。

#### 4.3.4 PeopleCount.py

用于人群的检测和计数，可以选择图片或者视频模式，默认加载ssd.py中设置的参数，检测时会弹出窗口显示，检测后也可存储检测后的视频。

### 4.4 MCNN

#### 4.4.1 train.py

用于训练模型，可以调整训练参数。

#### 4.4.2 test.py

可加载已经训练好的权重进行图片上人群的计数，修改image_paths里的内容即可选择要预测的图片。

#### 4.4.3 PeopleCount.py

可加载已经训练好的权重进行图片上人群的计数，生成的密度图保存在imgout中。

## 五、参考资料

- [ultralytics/ultralytics: Ultralytics YOLO11 🚀](https://github.com/ultralytics/ultralytics)
- [项目首页 - faster-rcnn-pytorch:这是一个faster-rcnn的pytorch实现的库，可以利用voc数据集格式的数据进行训练。 - GitCode](https://gitcode.com/gh_mirrors/fa/faster-rcnn-pytorch?utm_source=csdn_blog_hover&isLogin=1)
- [项目首页 - ssd-pytorch:这是一个ssd-pytorch的源码，可以用于训练自己的模型。 - GitCode](https://gitcode.com/gh_mirrors/ssdp/ssd-pytorch?utm_source=csdn_github_accelerator&isLogin=1)
- [svishwa/crowdcount-mcnn: Single Image Crowd Counting via MCNN (Unofficial Implementation)](https://github.com/svishwa/crowdcount-mcnn?tab=readme-ov-file)
- [YOLOv11超详细环境搭建以及模型训练（GPU版本）-CSDN博客](https://blog.csdn.net/2401_85556416/article/details/143378148?ops_request_misc=&request_id=&biz_id=102&utm_term=yolov11环境配置&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-143378148.142^v101^pc_search_result_base6&spm=1018.2226.3001.4187)
- [目标检测：YOLOv11(Ultralytics)环境配置，适合0基础纯小白，超详细_yolov11环境配置-CSDN博客](https://blog.csdn.net/qq_67105081/article/details/143270109?ops_request_misc=%7B%22request%5Fid%22%3A%22614da8984610eb1db9898a7652e598fe%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=614da8984610eb1db9898a7652e598fe&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-143270109-null-null.142^v101^pc_search_result_base6&utm_term=yolov11环境配置&spm=1018.2226.3001.4187)
- [睿智的目标检测27——Pytorch搭建Faster R-CNN目标检测平台-CSDN博客](https://blog.csdn.net/weixin_44791964/article/details/105739918)
- [【深度学习】使用FasterRCNN模型训练自己的数据集（记录全流程_fasterrcnn训练自己的数据集-CSDN博客](https://blog.csdn.net/qq_53959016/article/details/142785760?ops_request_misc=&request_id=&biz_id=102&utm_term=fasterrcnn训练自己模型&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-142785760.nonecase&spm=1018.2226.3001.4187)
- [神经网络学习小记录-番外篇——常见问题汇总_loading weights into state dict... killed-CSDN博客](https://blog.csdn.net/weixin_44791964/article/details/107517428)
- [总结该问题解决方案：OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized-CSDN博客](https://blog.csdn.net/peacefairy/article/details/110528012)
- [关于OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.错误解决方法 - 知乎](https://zhuanlan.zhihu.com/p/371649016)
- [Python AttributeError: module ‘distutils‘ has no attribute ‘version‘_attributeerror: module 'distutils' has no attribut-CSDN博客](https://blog.csdn.net/Alexa_/article/details/132686602)
- [AttributeError: ‘ImageDraw‘ object has no attribute ‘textsize‘ 解决方案 - 知乎](https://zhuanlan.zhihu.com/p/676557745)
- [详细！正确！COCO数据集（.json）训练格式转换成YOLO格式（.txt）_coco数据集的train.txt-CSDN博客](https://blog.csdn.net/qq_42012888/article/details/120270283)
- [YOLO11改进|注意力机制篇|添加GAM、CBAM、CA、ECA等注意力机制_gam注意力模块-CSDN博客](https://blog.csdn.net/A1983Z/article/details/142671275)
- [[MCNN\] Crowd Counting 人群计数 复现过程记录_mcnn复现-CSDN博客](https://blog.csdn.net/wpw5499/article/details/106231707)
- [修改成清华镜像源解决Anaconda报The channel is not accessible源通道不可用问题_the channel is not accessible or is invalid.-CSDN博客](https://blog.csdn.net/fullbug/article/details/121561045)
