'''
基于dsvl的方案改动：
1. 舍弃mmca，因为这个改动了大模型内部结构，直接将大模型训练好的语言能力大幅降低
2. 图像层方面，使用llava的方案，抽取图像encode倒数第二层向量，线性映射层采用两层方式，并增加LayerNorm层
3. 图文concat环节中，dsvl是attention_mask矩阵中1表示文字部分，2表示图像部分，实际验证时发现不怎么work[应该是需要结合mmca]，这部分依旧还原到全部用1表示输入
4. 图文concat之后，embedding根据position_ids加了一次位置向量，这个操作略显奇怪，因为embedding进入大模型后会加上旋转位置向量的；等于它的这个模型加了两次位置向量，然后generate操作时进入模型仅仅只加了一次位置向量
5. 不改变原生大模型的结构之后，可以使用flash_attention加速训练
6. generate部分，因为用了原始的大语言模型，所以在输入时接受纯文本输入
'''

# deepspeed --master_addr 172.xx.94 --master_port 5050 --include localhost:0,1,2,3,4,5,6,7  Nlp_2023/Multi-Modal-Model/Visualmodel_Sft.py
import os

import torch
from datasets import load_dataset
from torch.utils.data import random_split
from transformers import TrainingArguments, Trainer, LlamaTokenizer
from transformers import CLIPImageProcessor
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from VisualModel.modeling_dsvl import create_dsvl_model_and_transforms

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)

para_dict = {
            "lm_model_name_or_path":"/Model_WinGPT_pretrain/Chinese_llama2_7B/epoch3/",
            "vision_model_name_or_path":"/data/public/clip-vit-large-patch14-336/",
            "use_cache":False,
            "use_flash_attention_2":False,
            "max_seq_len":4096,
            "vis_proj":"baseline_2"
            }

IGNORE_INDEX = -100


tokenizer = LlamaTokenizer.from_pretrained(para_dict['lm_model_name_or_path'])
image_processor = CLIPImageProcessor.from_pretrained(para_dict['vision_model_name_or_path'])


#add special token
special_token_list = ["<|image|>", "<|im_start|>", "<|im_end|>"] # used for easy image # replacement
tokenizer.add_tokens(special_token_list, special_tokens=True)

print(len(tokenizer))


#llama
START_IDS = [13, 4007, 22137, 29901]
END_IDS = tokenizer.eos_token_id
padding_id = tokenizer.pad_token_id

print([tokenizer.decode(START_IDS)])
print(tokenizer.decode(END_IDS))
print(tokenizer.decode(padding_id))


def tokenize(text):
    image_list = text['image_list']
    inputs_with_mask = tokenizer(text['text'])
    inputs = inputs_with_mask['input_ids']
    labels = [-100] * len(inputs)
    
    for i in range(len(labels)):
        if inputs[i - len(START_IDS): i]  == START_IDS:
            j = inputs.index(END_IDS, i)
            for k in range(i,j+1):
                labels[k] = inputs[k]

    return dict(
        images=image_list,
        input_ids=inputs,
        attention_mask=inputs_with_mask['attention_mask'],
        labels=labels,
        image_num=len(image_list),
        )

'''
data_process

data_type:
{'text':'data_text','image_list':[]}
'''

instruction_dataset = load_dataset("json", data_files="/Nlp_2023/Multi-Modal-Model/data/visual_data.json", split="train", cache_dir="/workspace/cache_dir/")
tokenized_dataset = instruction_dataset.map(tokenize, remove_columns=instruction_dataset.column_names, num_proc=32, keep_in_memory=False)

train_size = int(0.99 * len(tokenized_dataset))
train_dataset, val_dataset = random_split(tokenized_dataset , [train_size, len(tokenized_dataset) - train_size])
print(len(train_dataset), len(val_dataset))


'''
model training
'''

training_args = TrainingArguments(
    output_dir='./unfronzen_multi-modal_results',   # output directory
    num_train_epochs=3,                            # total number of training epochs
    per_device_train_batch_size=6,                # batch size per device during training
    per_device_eval_batch_size=6,                 # batch size for evaluation
    warmup_steps=50,                              # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                             # strength of weight decay
    evaluation_strategy="epoch",                   # Evaluation is done at the end of each epoch
    logging_steps=30,
    save_strategy='epoch',
    learning_rate=1e-5,
    gradient_accumulation_steps=3,
    bf16=True,
    gradient_checkpointing=True,
    deepspeed='./ds_config_sft.json'
    )


model, _ = create_dsvl_model_and_transforms(text_tokenizer=tokenizer, args=para_dict)

print('model_loaded')

def concat(batch):
    img_tensor_list = []
    for k in batch:
        for img in k['images']:
            tmp_image = image_processor.preprocess(Image.open(img).convert('RGB'))['pixel_values'][0]
            img_tensor_list.append(torch.tensor(tmp_image).unsqueeze(0).half())
    return img_tensor_list

def the_collate_fn(batch): 
    input_ids = pad_sequence([torch.tensor(f['input_ids']) for f in batch], padding_value=padding_id, batch_first=True)
    images_ids = torch.cat(concat(batch), dim=0) 
    attention_mask = pad_sequence([torch.tensor(f['attention_mask']) for f in batch], padding_value=0, batch_first=True)
    labels = pad_sequence([torch.tensor(f['labels'])  for f in batch], padding_value=IGNORE_INDEX, batch_first=True)
    image_num = [f['image_num'] for f in batch]
    return {'images':images_ids, 'input_ids':input_ids,  'attention_mask':attention_mask, 'labels':labels, 'img_num':image_num}


class Mytrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(images=inputs["images"], input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels = inputs["labels"], image_num=inputs["img_num"],)
        loss, logits = outputs[:2]
        return (loss, logits) if return_outputs else loss

trainer = Mytrainer(model=model, 
                    args=training_args, 
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset, 
                    data_collator=the_collate_fn,
                    # callbacks=[MetricsCallback()]
                    )

trainer.train()
