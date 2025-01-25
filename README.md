<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> Convtimenet: 一种用于多变量时间序列分析的深度分层全卷积模型 (ACM WWW2025, 已接受) </b></h2>
</div>

测试是否成功提交
---
>
> 🙋 如果您发现错误或有任何建议，请告诉我们！
> 
> 🌟 如果您觉得这个资源有帮助，请考虑为此仓库加星并引用我们的研究：

```
@article{cheng2024convtimenet,
  title={Convtimenet: A deep hierarchical fully convolutional model for multivariate time series analysis},
  author={Cheng, Mingyue and Yang, Jiqian and Pan, Tingyue and Liu, Qi and Li, Zhi},
  journal={arXiv preprint arXiv:2403.01493},
  year={2024}
}
```

## 项目概述

这是 ConvTimenet 的官方开源代码。

论文链接：[ConvTimeNet: A Deep Hierarchical Fully Convolutional Model for Multivariate Time Series Analysis](https://arxiv.org/abs/2403.01493)

## 关于 ConvTimeNet

在这项研究中，我们深入探讨了如何重新激活卷积网络在时间序列分析建模中的作用。

![image](https://github.com/Mingyue-Cheng/ConvTimeNet/assets/75526778/6ad67b14-ec3a-42b4-970f-585108a13bd6)

ConvTimeNet 是一种深度分层全卷积网络，可以作为时间序列分析的多功能骨干网络。ConvTimeNet 的一个关键发现是，保留一个深度和分层的卷积网络，并配备现代技术，可以在性能上优于或与流行的 Transformer 网络和开创性的卷积模型相媲美。对时间序列的预测和分类进行的大量实验充分证明了其有效性。总体而言，我们希望 ConvTimeNet 能作为一种替代模型，并鼓励研究界重新思考卷积在时间序列挖掘任务中的重要性。

## 可变形补丁嵌入

![image](https://github.com/Mingyue-Cheng/ConvTimeNet/assets/75526778/115bd0cd-c011-468e-b305-12526e773225)

可变形补丁嵌入在 ConvTimeNet 的性能中起着至关重要的作用，通过自适应调整补丁的大小和位置，巧妙地对时间序列数据进行标记。

## ConvTimeNet 块

![ConTimeNet_backbone](https://github.com/Mingyue-Cheng/ConvTimeNet/assets/75526778/5ee724c0-3956-492a-9601-82a235ed7ffc)

ConvTimeNet 块有三个关键设计：(1) 不同大小的卷积核，以捕捉不同时间尺度的特征。(2) 可学习的残差，使网络更深。(3) 深度卷积，与普通卷积相比，计算量更少，提高了模型的效率。

## 主要结果

我们的 ConvTimeNet 在时间序列分类任务和时间序列长期预测任务中达到了 SOTA 的效果。
![cd-diag](https://github.com/Mingyue-Cheng/ConvTimeNet/assets/75526778/d1ef9c1a-2d0a-4c91-b02c-6390221868b3)

![radar](https://github.com/Mingyue-Cheng/ConvTimeNet/assets/75526778/51cd735d-cee0-413f-8f49-d97e5334f367)
## 检查环境
在终端中依次输入：

```bash
pip show matplotlib
pip show numpy
pip show pandas
pip show scikit-learn
pip show thop
pip show torch
```

## 安装

1. 安装依赖项。pip install -r requirements.txt

也可以手动安装

```bash
pip install matplotlib==3.8.1
pip install numpy==1.26.4
pip install pandas==2.2.1
pip install scikit_learn==1.3.2
pip install thop==0.1.1.post2209072238
pip install torch==2.0.1
```

2. 下载数据。您可以从 [AutoFormer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) 下载所有数据集。对于每个 csv 文件，创建一个文件夹 ./dataset/{csv_name} 并将 csv 文件放入其中。

3. 训练。所有脚本都在目录 ./scripts 中。请阅读 ./run_longExp.py 以了解脚本中参数的含义。

## 测试数据
现在，我已经安装好相关的包了，我准备了一份数据，数据的路径为 `.\dataset\weather\weather.csv`


### 1. 确保数据格式正确
首先，确保您的数据文件 weather.csv 格式正确，并且与模型预期的输入格式一致。通常，数据文件应包含时间戳和特征列。

### 

2. 更新配置文件
在运行模型之前，您需要更新配置文件以指向您的数据集路径。假设您使用的是 `run_longExp.py` 脚本，您需要确保配置文件中的路径参数正确设置。例如：

```yaml
root_path: "E:/浏览器下载地址/宁海疾控食源性疾病/TimeNet/dataset/weather/"
data_path: "weather.csv"
```
### 



### 3. 修改数据加载器
如果您的数据格式与现有的数据加载器不完全匹配，您可能需要修改 `data_provider/data_loader.py` 中的 `Dataset_Custom` 类，以适应您的数据格式。

### 4. 运行测试脚本
在确保配置文件和数据加载器正确后，您可以运行测试脚本。假设您使用的是 `run_longExp.py`，可以在命令行中执行以下命令：

```bash
python run_longExp.py --config your_config_file.yaml --mode test
```

确保将 `your_config_file.yaml` 替换为您的配置文件。

### 5. 检查输出
运行完成后，检查模型的输出结果。根据需要，您可以调整模型参数或数据预处理步骤以获得更好的性能。


## 引用

```
@article{cheng2024convtimenet, 
  title={Convtimenet: A deep hierarchical fully convolutional model for multivariate time series analysis}, 
  author={Cheng, Mingyue and Yang, Jiqian and Pan, Tingyue and Liu, Qi and Li, Zhi}, 
  journal={arXiv preprint arXiv:2403.01493}, 
  year={2024} 
}
```

### 进一步阅读

1, [**FormerTime: Hierarchical Multi-Scale Representations for Multivariate Time Series Classification**](https://arxiv.org/pdf/2302.09818).

**作者**: Cheng, Mingyue and Liu, Qi and Liu, Zhiding and Li, Zhi and Luo, Yucong and Chen, Enhong

```bibtex
@inproceedings{cheng2023formertime,
  title={Formertime: Hierarchical multi-scale representations for multivariate time series classification},
  author={Cheng, Mingyue and Liu, Qi and Liu, Zhiding and Li, Zhi and Luo, Yucong and Chen, Enhong},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={1437--1445},
  year={2023}
}
```

2, [**InstructTime: Advancing Time Series Classification with Multimodal Language Modeling**](https://arxiv.org/pdf/2403.12371).

**作者**: Cheng, Mingyue and Chen, Yiheng and Liu, Qi and Liu, Zhiding and Luo, Yucong

```bibtex
@article{cheng2024advancing,
  title={Advancing Time Series Classification with Multimodal Language Modeling},
  author={Cheng, Mingyue and Chen, Yiheng and Liu, Qi and Liu, Zhiding and Luo, Yucong},
  journal={arXiv preprint arXiv:2403.12371},
  year={2024}
}
```

3, [**TimeMAE: Self-supervised Representation of Time Series with Decoupled Masked Autoencoders**](https://arxiv.org/pdf/2303.00320).

**作者**: Cheng, Mingyue and Liu, Qi and Liu, Zhiding and Zhang, Hao and Zhang, Rujiao and Chen, Enhong

```bibtex
@article{cheng2023timemae,
  title={Timemae: Self-supervised representations of time series with decoupled masked autoencoders},
  author={Cheng, Mingyue and Liu, Qi and Liu, Zhiding and Zhang, Hao and Zhang, Rujiao and Chen, Enhong},
  journal={arXiv preprint arXiv:2303.00320},
  year={2023}
}
```

4, [**CrossTimeNet: Learning Transferable Time Series Classifier with Cross-Domain Pre-training from Language Model**](https://arxiv.org/pdf/2403.12372).

**作者**: Cheng, Mingyue and Tao, Xiaoyu and Liu, Qi and Zhang, Hao and Chen, Yiheng and Lei, Chenyi

```bibtex
@article{cheng2024learning,
  title={Learning Transferable Time Series Classifier with Cross-Domain Pre-training from Language Model},
  author={Cheng, Mingyue and Tao, Xiaoyu and Liu, Qi and Zhang, Hao and Chen, Yiheng and Lei, Chenyi},
  journal={arXiv preprint arXiv:2403.12372},
  year={2024}
}
```






