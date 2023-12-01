# Large-Multimodal-Model

### 本项目参考
[DeepSpeed-VisualChat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-VisualChat)

### 主要改动
```
基于dsvl的方案改动：
1. 舍弃mmca，因为这个改动了大模型内部结构，直接将大模型训练好的语言能力基本归0
2. 图文concat环节中，dsvl是attention_mask矩阵中1表示文字部分，2表示图像部分，实际验证时发现不怎么work[应该是需要结合mmca]，这部分依旧还原到全部用1表示输入
3. 图文concat之后，embedding根据position_ids加了一次位置向量，这个操作略显奇怪，因为embedding进入大模型后会加上旋转位置向量的；等于它的这个模型加了两次位置向量，然后generate操作时进入模型仅仅只加了一次位置向量
4. 不改变原生大模型的结构之后，可以使用flash_attention加速训练
5. generate部分，因为用了原始的大语言模型，所以在输入时接受纯文本输入
```
