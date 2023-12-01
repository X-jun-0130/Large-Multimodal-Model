import json
import random
import os
from tqdm import tqdm

end_token = '</s>'
sepcial_token = '<image>' #使用special-token在文本指令中代替图片的位置


'''
text_data

图文模型训练后，原语言模型的生成能力会极大的丧失，增加高质量文本指令，使其训练后语言能力得以部分恢复。
#为了增加文字指令的泛化性: 
1.对没有标点的指令随机进行添加，有标点的随机去除；
2.对有换行符的指令，进行随机去除，
3.随机概率为1/5
'''

def get_random_insert_punctuation(_line): 
    punc_list = ['.', '。', '?', '？', ';', '；', '，', ',']
    p = random.choice([1,2,3,4,5])

    if _line[-1] not in punc_list:
        if p == 1:
            _line = _line + random.choice(punc_list)
    else:
        if p == 2:
            _line = _line[:-1]

    if '\n' in _line:
        if p == 5:
            _line = _line.replace('\n', '')

    return _line


def get_text_instructions(ins_list):
    instructions_list = []
    for key in ins_list:
        if len(key['instruction']) > 1 and len(key['output'][0]) >= 1 and key['output'][0] != "":
            if len(key['input']) > 0:
                ins_input = key['instruction'] + '\n' +key['input'] 
                ins_out = str(key['output'][0]).strip(' ')  +end_token
            else:
                ins_input = key['instruction'] 
                ins_out = str(key['output'][0]).strip(' ') + end_token

            line = 'User: '+ get_random_insert_punctuation(ins_input)+ end_token + '\n Assistant: '+ ins_out

            instructions_list.append(line)
    return instructions_list

def get_mt(ins_list, max_len=600):
    instructions_list = []
    for mt in ins_list:
        if 'User:' in mt[-1][:6]:
            mt = mt[:-1]
        chunk_token = end_token+'\n '

        dd_s = chunk_token.join(mt) + end_token

        if len(dd_s) <= max_len:
            pass
        else:
            while len(dd_s) > max_len:
                mt = mt[:-2]
                dd_s = chunk_token.join(mt) + end_token
        dd_s = dd_s.replace('User:', 'User: ').replace('\n Assistant:', '\n Assistant: ')
        instructions_list.append(dd_s)
    return instructions_list

winning_clean_v2 = [json.loads(k)['output'] for k in open('./sft/multi_turn/clean_v2.jsonl', 'r', encoding='utf-8')]
multi_data = [k for k in get_mt(winning_clean_v2, 800)]


universal_single_filelist = ['./sft/single_turn/', './med_sft/single_turn/']
text_instructions = []

for universal_single_file in universal_single_filelist:
    single_universal_ins = []
    for single_file in os.listdir(universal_single_file):
        print(single_file)
        single_path = universal_single_file + single_file
        single_ins = [json.loads(k) for k in open(single_path, 'r', encoding='utf-8')]

        if 'alpaca_gpt4_data_en' in single_file or 'webgpt' in single_file or 'rain3.5MandOpenorca_select_dedup' in single_file:
            pass
        else:
            single_universal_ins.extend(single_ins)

    random.shuffle(single_universal_ins)
    text_instructions += [k for k in get_text_instructions(single_universal_ins) if len(k) < 1000  and '<image>' not in k]

text_instructions += multi_data
random.shuffle(text_instructions)
print(len(text_instructions))


def delete_space(strs): #去除一些中文文字之间产生的错误空格
    sss = strs.split(' ')
    sss = [k for k in sss if k != '']
    if len(sss) >= 2:
        kkk = []
        for i in range(len(sss)-1):
            if sss[i][-1].encode('utf-8').isalpha() and sss[i+1][0].encode('utf-8').isalpha():
                kkk.append(sss[i])
                kkk.append(' ')
            else:
                kkk.append(sss[i])
        kkk.append(sss[-1])
        tt = ''.join(kkk)
    else:
        tt = strs
    return tt

def convert_to_target_format(text, image_list):
    # Initialize the target format dictionary
    target_data = {
        "text": text, 
        "image_list": image_list, 
    }
    return target_data

def get_input(image_instructions):
    _instructyions = []
    #这部分处理的是单幅图片的场景
    for key in image_instructions:
        try:
            image_f = key['image_file']
            image_p = image_dict[image_f][key['image']]
            image_list = [image_p]
            conversation = key['conversations']
            k_list = []
            for k in conversation:
                k_list.append(k['from'] + ':' + delete_space(k['value']))
            
            end_tokens = end_token +'\n '
            line = end_tokens.join(k_list)
            line = line.replace('human:', 'User: ')
            line = line.replace('gpt:', 'Assistant: ')
            line = line.replace('<image>', '<im_start><image><im_end>')
            _instructyions.append([line + end_token, image_list])
        except:
            pass
    
    return _instructyions
'''
image-text-data

读取各图片文件，处理成地址信息
'''

image_path = './Images/'
image_dict ={}

for k in os.listdir(image_path):
    image_file = image_path + k + '/'
    image_dict[k] = {j:image_file+j for j in os.listdir(image_file)}

image_instructions = []
json_path = './Nlp_2023/Visual_Data/instructions/'
for jf in os.listdir(json_path):
    json_list = json.load(open(json_path + jf, 'r', encoding='utf-8'))
    random.shuffle(json_list)
    if 'chinese_text_recognition' in jf:
        image_instructions += get_input(json_list[:150000])
    else:
        image_instructions += get_input(json_list)
    print(jf, str(len(json_list)), str(len(image_instructions)))

random.shuffle(image_instructions)
print(len(image_instructions))
    
data_all = []

for x in range(len(image_instructions)):
    line, image_list = image_instructions[x]
    # 将纯文本指令拼接到图文指令中
    if x < len(text_instructions):
        text_ins = text_instructions[x]
    else:
        text_ins = ''

    if text_ins != '':
        _list = [line, text_ins]
        random.shuffle(_list)
        concat_line = _list[0] + '\n ' + _list[1]
    else:
        concat_line = line

    data_all.append([concat_line, image_list])

from joint_data import get_chunk,get_length

datalist = get_length(data_all)

#拼接到接近max_length的长度
new_data_all = get_chunk(datalist, 2048)

print(len(new_data_all))
print(new_data_all[1000])

with open('./data/visual_data_instructions.json', 'a', encoding='utf-8') as target_file:
    for i in tqdm(range(len(new_data_all))):
        item = new_data_all[i]
        if item[0].count(sepcial_token) == len(item[1]): #判断处理数据是否正确
            _t = convert_to_target_format(item[0], item[1])
            target_file.write(json.dumps(_t, ensure_ascii=False) + '\n')
#最终指令形式：{"text": "User: <im_start><image><im_end>\n这个肿块的位置在哪里？请根据图中内容简洁作答。</s>\n Assistant: 左侧中央区域。</s>\n User: <im_start><image><im_end>\n标题中所指的是哪一颗牙齿？请尽可能简洁地回答。</s>\n Assistant: 左上第一臼齿。</s>\n User: 图片中庆祝的主要事件是什么?\n<im_start><image><im_end></s>\n Assistant: 该图中庆祝的主要事件是美国陆军的生日,如旗饼和军人疲劳的出现所显示。</s>\n User: 蛋糕是什么样子?</s>\n Assistant: 蛋糕與美國國旗相似,暗示了美國陸軍的主題。</s>\n User: 参加这个活动有成年人和孩子吗?</s>\n Assistant: 是的,这张照片捕捉了各种年龄的人们,包括成年人和孩子,参加这个活动。</s>\n User: 那个疲倦的人在蛋糕旁边做什么?</s>\n Assistant: 疲劳的人朝着蛋糕伸出手来,可能表明他正在切蛋糕或提供蛋糕。</s>\n User: 庆祝活动期间,客人如何安排在桌旁?</s>\n Assistant: 客人正站在一张用红布覆盖的长桌旁,他们似乎聚集来见证蛋糕切或服务的过程。</s>\n User: 使用了什么类型的显微镜来捕捉这些图像？请根据图中内容尽量简洁作答。\n<im_start><image><im_end></s>\n Assistant: 荧光显微镜。</s>\n User: <im_start><image><im_end>\nCan you describe the main features of this image for me?</s>\n Assistant: The image captures a serene day at the Citrus Heights Water District,located at6230.The building itself is a single-story structure,painted in a calming shade of beige.Its design is modern and welcoming,with a curved entrance that adds a unique architectural element.\n\nTwo large orange spheres flank the entrance,their vibrant color contrasting beautifully with the building's beige exterior.These spheres,along with three others scattered across the parking lot,add a playful touch to the scene.\n\nThe American flag,flying at half-mast on a flagpole,adds a solemn note to the otherwise cheerful setting.The blue sky overhead is clear and bright,hinting at good weather.In the background,trees can be seen,their green foliage adding a touch of nature to the urban landscape.\n\nOverall,the image presents a harmonious blend of architecture and nature,underlined by the striking orange spheres that dot the premises.</s>", "image_list": ["/workspace/Xuxiangjun/Images/pmc_vqa/PMC2894136_pntd-0000726-g005_67838.jpg", "/workspace/Xuxiangjun/Images/pmc_vqa/PMC7204013_Fig2_164782.jpg", "/workspace/Xuxiangjun/Images/coco-2017/000000258882.jpg", "/workspace/Xuxiangjun/Images/pmc_vqa/PMC3440378_pone-0043463-g003_154434.jpg", "/workspace/Xuxiangjun/Images/coco-2017/000000199669.jpg"]}
