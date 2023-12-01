from concurrent.futures import ProcessPoolExecutor
import random
from transformers import  LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained('/Model_FT/WiNGPT-002-0819/')
special_token_list = ["<image>", "<im_start>", "<im_end>"] # used for easy image # replacement
tokenizer.add_tokens(special_token_list, special_tokens=True)


def mp_process(function,txt_list):
    select_all = []
    with ProcessPoolExecutor(max_workers=32) as Pool:
        results = Pool.map(function, txt_list)
        for result in results:
            select_all.extend(result)
    return select_all

def get_tokenizer(datalist):
    data_token = []
    for k in datalist:
        token_leng = len(tokenizer.encode(k[0])) + len(k[1])*256
        data_token.append([k[0], token_leng, k[1]])
    return data_token

def get_length(data_list):
    x = 32
    y = int(len(data_list)/x)+1
    x_all = []
    for i in range(x):
        slice_list = data_list[i*y:(i+1)*y]
        x_all.append(slice_list)

    tokenizer_list = mp_process(get_tokenizer, x_all)
    return tokenizer_list


def joint(txt_list, maxlen):
    new_txtlist = []
    point = set()
    for i, (text, length, images) in enumerate(txt_list):
        if i in point:
            continue
        joint_str = text
        length_joint = length
        image_list = images
        point.add(i)
        for j in range(i+1, len(txt_list)):
            if j in point:
                continue
            new_length = length_joint + txt_list[j][1] + 1
            if maxlen-30 <= new_length <= maxlen:
                joint_str += '\n ' + txt_list[j][0]
                image_list += txt_list[j][2]
                point.add(j)
                break
            elif new_length < maxlen-30:
                joint_str += '\n ' + txt_list[j][0]
                length_joint = new_length
                image_list += txt_list[j][2]
                point.add(j)
        new_txtlist.append([joint_str, image_list])
    return new_txtlist

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

def get_chunk(data_tokenizer, max_token):
    data_list = []
    random.shuffle(data_tokenizer)
    
    # 直接添加大于等于max_token的元素
    _list=[[item[0], item[2]] for item in data_tokenizer if item[1] > max_token]
    print(len(_list))
    
    # 筛选小于max_token的元素
    small_token_items = [item for item in data_tokenizer if item[1] <= max_token]
    total_items = len(small_token_items)
    
    # 分成100批进行组合
    num_slices = 150
    items_per_slice = int(total_items / num_slices) + 1
    
    for index in tqdm(range(num_slices)):
        start = index * items_per_slice
        end = min((index + 1) * items_per_slice, total_items)
        slice_list = small_token_items[start:end]
        data_list.extend(joint(slice_list, max_token))

    # def process_slice(index):
    #     start = index * items_per_slice
    #     end = min((index + 1) * items_per_slice, total_items)
    #     slice_list = small_token_items[start:end]
    #     return joint(slice_list, max_token)
    
    # with ProcessPoolExecutor() as executor:
    #     future_to_slice = {executor.submit(process_slice, i): i for i in range(num_slices)}
    #     for future in tqdm(concurrent.futures.as_completed(future_to_slice), total=num_slices):
    #         data_list.extend(future.result())
    
    return data_list
