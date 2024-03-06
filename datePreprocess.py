import pandas as pd
import numpy as np
from tqdm._tqdm_notebook import tqdm_notebook
from transformers import BertTokenizer
from repalceList import replace_list
from keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# 1.Dtae processing
# 1.1 import data
df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('test.csv')
df_val=pd.read_csv('validation.csv')
df_all = pd.concat([df_train,df_test,df_val])
# 1.2 date preprocess using BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def bert_encode(data,maximum_length) :
    input_ids = []
    attention_masks = []
    for i in tqdm_notebook(range(len(data))):
        encoded = tokenizer.encode_plus(
        text=data[i],
        add_special_tokens=True,
        max_length=maximum_length,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True,
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)

def replace_abbreviations(text):
    for k, v in replace_list.items():
        text = text.replace(k, v)
    return text
df_all['text'] = df_all['text'].apply(replace_abbreviations)
maximum_length=df_all['text'].apply(lambda x: len(x.split(" "))).max()

input_ids,attention_masks=bert_encode(df_all.text.values,maximum_length)