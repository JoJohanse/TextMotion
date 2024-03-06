import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer,TFBertModel
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from sklearn.metrics import accuracy_score
from keras.metrics import CategoricalAccuracy
from transformers import TFBertModel
from keras.callbacks import EarlyStopping
from datePreprocess import df_all

# load model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

tokenizer.save_pretrained('bert_tokenizer')
bert_model.save_pretrained('bert_model')

train_set, combine_set = train_test_split(df_all, test_size=0.2, random_state=42, stratify=df_all['label'])
test_set, val_set = train_test_split(combine_set, test_size=0.5, random_state=42, stratify=combine_set['label'])

trainer = tokenizer(
    text=train_set['text'].to_list(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True
)

tester = tokenizer(
    text=test_set['text'].to_list(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True,
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True
)

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

max_length = 70

input_ids = Input(shape=(max_length,), name='input_ids', dtype=tf.int32)
input_mask = Input(shape=(max_length,), name='attention_mask', dtype=tf.int32)
embedding = bert_model(input_ids, attention_mask=input_mask)[0]

out = tf.keras.layers.GlobalMaxPool1D()(embedding)
out = tf.keras.layers.BatchNormalization()(out)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32, activation='relu')(out)
results = Dense(6, activation='sigmoid')(out)

model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=[results])
model.layers[2].trainable = True

optimizer = Adam(
    learning_rate=5e-5,
    decay=0.01,
    epsilon=1e-08,
    clipnorm=1.0
)

loss = CategoricalCrossentropy(from_logits=True)
metric = CategoricalAccuracy('accuracy')

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric]
)

early_stopping = EarlyStopping(monitor='val_loss', patience=4)

train_history = model.fit(
    x={'input_ids': trainer['input_ids'], 'attention_mask': trainer['attention_mask']},
    y=to_categorical(train_set['label']),
    validation_data=({'input_ids': tester['input_ids'], 'attention_mask': tester['attention_mask']}, to_categorical(test_set['label'])),
    epochs=10,
    batch_size=36,
    callbacks=[early_stopping],
    verbose= 1
)
prediction = model.predict({'input_ids': valer['input_ids'], 'attention_mask': valer['attention_mask']})
print(prediction)
prediction_np = np.argmax(prediction, axis=1)
accuracy_val = accuracy_score(val_set['label'], prediction_np)
# 输出预测各个情绪的可能性
print('accuracy_val', accuracy_val)

model.save('trained_model/bert_fine_tune.h5')

# Plot loss and accuracy
epochs = range(1, len(train_history.history['loss']) + 1)

plt.plot(epochs, train_history.history['loss'])
plt.plot(epochs, train_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')

plt.plot(epochs, train_history.history['accuracy'])
plt.plot(epochs, train_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')

# 预测测试
for i in range(10):
    print('prediction:', val_set['text'].to_list()[i])
    print('anger',prediction[i][0]*100, '%')
    print('fear',prediction[i][1]*100, '%')
    print('joy',prediction[i][2]*100, '%')
    print('love',prediction[i][3]*100, '%')
    print('sadness',prediction[i][4]*100, '%')
    print('surprise',prediction[i][5]*100, '%')