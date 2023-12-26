# Large-Multimodal-Model

### 本项目参考
[DeepSpeed-VisualChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-VisualChat)

[LlaVA](https://github.com/haotian-liu/LLaVA)

### 一、模型方案
模型基于DSVL以及llava两方案进行改动，算法整体上接近llava，由于llava的实现仅支持单图，针对这一问题利用DSVL的图文拼接方案进行补充；
训练时，分为两阶段：1.只训练投影层(两个线性层+LayerNorm)；2.冻结图像层，微调投影层和语言模型层
[模型架构](https://github.com/X-jun-0130/Large-Multimodal-Model/blob/main/example/model.jpg)

### 二、主要改动
```
基于DSVL+llava方案改动：
1. 舍弃mmca，因为这个改动了大模型内部结构，直接将大模型训练好的语言能力大幅降低
2. 图像层方面，使用llava的方案，抽取图像encode倒数第二层向量，线性映射层采用两层方式，并增加LayerNorm层
3. 图文concat环节中，dsvl是attention_mask矩阵中1表示文字部分，2表示图像部分，实际验证时发现不怎么work[应该是需要结合mmca]，这部分依旧还原到全部用1表示输入
4. 图文concat之后，embedding根据position_ids加了一次位置向量，这个操作略显奇怪，因为embedding进入大模型后会加上旋转位置向量的；等于它的这个模型加了两次位置向量，然后generate操作时进入模型仅仅只加了一次位置向量
5. 不改变原生大模型的结构之后，可以使用flash_attention加速训练
6. generate部分，因为用了原始的大语言模型，所以在输入时接受纯文本输入
```

### 三、模型优劣势
- 优势
  - 采用dsvl的图文拼接方案，所以支持任意的图片输入，如：多轮多图，多轮单图等等
  - 语言模型的能力得到大幅保留，并且实际使用时，支持纯文本的输入输出，理论上一个模型真正的做到文本模型与多模态模型
  - 理论上支持其他非llama语言模型(待测试)

- 劣势
  - 模型整体代码方案基于DSVL进行改动而来，在推理时仍需要加载初始预训练大语言模型，部署很不方便 
  

### 四、实现细节
- 数据处理
  - 仅图文数据训练会使语言模型原语言能力遭受遗忘，故在制作数据集时，增加了一定数量的文本指令集
  - 增加三个special-token，来定义图片在文本中的图片信息
  - 将图文指令与文本指令拼接到一定长度(2048token)进入模型进行训练
  
- 模型
  - 采用llava的图像层方案抽取vis-encoder倒数第二层信息进入线性映射层
  - 线性映射层采用两层，并增加LayerNorm提高模型训练时的稳定型
  - 大语言模型方面不做任何的变动
      
- 模型微调
  - 使用transformers中的trainer模型重写了微调的代码，使用deepspeed进行训练
  - 训练3个epoch，初始lr=3e-5，实际中发现lr可能过大，第二个epoch就发现了过拟合

- 测试发现
  - 模型对于简单的花花草草的识别，倒是问题不大，但是ocr能力较难训练
  - 目前高质量的中文图文数据集几乎没有，使用的llava翻译的语料，训练的模型幻觉问题明显
  - 需要制作图文不符的数据集来增加模型的泛化性


### 五、Examples

<details>
<summary>示例1-多图识别</summary>
<img src="https://github.com/X-jun-0130/Large-Multimodal-Model/blob/main/example/example1.jpg"/> 
</details>

<details>
<summary>示例2-多轮对话</summary>
<img src="https://github.com/X-jun-0130/Large-Multimodal-Model/blob/main/example/example2.jpg"/> 
</details>

<details>
<summary>示例3-多图识别</summary>
<img src="https://github.com/X-jun-0130/Large-Multimodal-Model/blob/main/example/example3.jpg"/> 
</details>
