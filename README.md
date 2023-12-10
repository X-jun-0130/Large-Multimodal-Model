# Large-Multimodal-Model

### 本项目参考
[DeepSpeed-VisualChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-VisualChat)
[LlaVA](https://github.com/haotian-liu/LLaVA)

### 模型方案
模型基于DSVL以及llava两方案进行改动，算法整体上接近llava，由于llava的实现仅支持单图，针对这一问题利用DSVL的图文拼接方案进行补充；
训练时，冻结图像层，仅微调线性映射(两层)和语言模型层

### 主要改动
```
基于DSVL+llava方案改动：
1. 舍弃mmca，因为这个改动了大模型内部结构，直接将大模型训练好的语言能力大幅降低
2. 图像层方面，使用llava的方案，抽取图像encode倒数第二层向量，线性映射层采用两层方式，并增加LayerNorm层
3. 图文concat环节中，dsvl是attention_mask矩阵中1表示文字部分，2表示图像部分，实际验证时发现不怎么work[应该是需要结合mmca]，这部分依旧还原到全部用1表示输入
4. 图文concat之后，embedding根据position_ids加了一次位置向量，这个操作略显奇怪，因为embedding进入大模型后会加上旋转位置向量的；等于它的这个模型加了两次位置向量，然后generate操作时进入模型仅仅只加了一次位置向量
5. 不改变原生大模型的结构之后，可以使用flash_attention加速训练
6. generate部分，因为用了原始的大语言模型，所以在输入时接受纯文本输入
```

### 模型优劣势
- 优势
  - 采用dsvl的图文拼接方案，所以支持任意的图片输入，如：多轮多图，多轮单图等等
  - 语言模型的能力得到大幅保留，并且实际使用时，支持纯文本的输入输出，理论上一个模型真正的做到文本模型与多模态模型
  
  
