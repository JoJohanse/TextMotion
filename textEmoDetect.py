import tensorflow as tf
from transformers import AutoTokenizer,TFBertModel
from transformers import TFBertModel
from training import val_set

# load model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
max_length = 70

model = tf.keras.models.load_model('trained_model/bert_fine_tune.h5', custom_objects={'TFBertModel': TFBertModel})
valer = tokenizer(
    text=val_set['text'].to_list(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True
)
prediction = model.predict({'input_ids': valer['input_ids'], 'attention_mask': valer['attention_mask']})
# 测试
for i in range(10):
    print('prediction:', val_set['text'].to_list()[i])
    print('anger',prediction[i][0]*100, '%')
    print('fear',prediction[i][1]*100, '%')
    print('joy',prediction[i][2]*100, '%')
    print('love',prediction[i][3]*100, '%')
    print('sadness',prediction[i][4]*100, '%')
    print('surprise',prediction[i][5]*100, '%')