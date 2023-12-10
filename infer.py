import os
import requests
from io import BytesIO
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from transformers import LlamaTokenizer
from PIL import Image
from VisualModel.modeling_dsvl import create_dsvl_model_and_transforms
from transformers import CLIPImageProcessor
import warnings
warnings.filterwarnings("ignore")

para_dict = {
            "lm_model_name_or_path":"/Model_WinGPT_pretrain/Chinese_llama2_7B/epoch3/",
            "vision_model_name_or_path":"/data/public/clip-vit-large-patch14-336/",
            "checkpoint_path":"/Model_WiNGPT_FT/multi-modal-llama7B/",
            "num_gpus":1,
            "device":'cuda',
            "max_gpu_memory":None,
            "use_flash_attention_2":False,
            "use_cache":False,
            "vis_proj":"baseline_2"
            }


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


tokenizer = LlamaTokenizer.from_pretrained(para_dict['lm_model_name_or_path'])
image_processor = CLIPImageProcessor.from_pretrained(para_dict['vision_model_name_or_path'])


#add special token
special_token_list = ["<|image|>", "<|im_start|>", "<|im_end|>"] # used for easy image # replacement
tokenizer.add_tokens(special_token_list, special_tokens=True)

tokenizer.padding_side = 'right'


'''
[{'image_path':[], 'user':'', 'assistant':''}, {}]
'''
def get_model_input(data_format):
    input_list = []
    image_list = []

    for k in data_format:
        image_list.extend(k['image_path'])
        n = len(k['image_path'])
        image_str = ''
        for _ in range(n):
            image_str += '<|image|>\n'
        
        line = 'User: '+image_str+k['user']+'</s>\n '+'Assistant:' + k['assistant']

        input_list.append(line)
    
    tmp_images = []
    for image_path in image_list:
        if image_path != '':
            image = load_image(image_path)
            tmp_image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half()
            tmp_images.append(tmp_image_tensor)

    if len(tmp_images) > 0:
        image_tensor = torch.cat(tmp_images, dim=0)
    else:
        image_tensor = None

    input_lint = '</s>\n '.join(input_list)
    input_lint = input_lint.replace('<|image|>', '<|im_start|><|image|><|im_end|>')
    print(input_lint)
    input_ids = tokenizer.encode(input_lint,return_tensors='pt')

    return image_tensor, input_ids


data_format = [{'image_path':['https://lmg.jj20.com/up/allimg/1114/0406210Z024/2104060Z024-5-1200.jpg'], 'user':'描述一下这副美丽的图片.', 'assistant':''},
               ]

image_tensor, input_ids = get_model_input(data_format)

model,  _  = create_dsvl_model_and_transforms(
                                            text_tokenizer = tokenizer,
                                            args=para_dict,
                                            )

model.load_state_dict(torch.load(os.path.join(para_dict['checkpoint_path'], 'pytorch_hf.bin'), map_location='cpu'), strict=False) 

model = model.eval()
model = model.to('cpu')

outputs = model.generate(images=image_tensor, input_ids=input_ids,  generation_kwargs={'repetition_penalty':1.1, "max_new_tokens":512})
print(outputs)
