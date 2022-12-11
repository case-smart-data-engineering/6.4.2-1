#!/usr/bin/env python3

from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import numpy as np
import sys

batch_size = 64  # Batch size for training.
epochs = 1000  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
data_path = '/workspace/6.5.2-1/1_算法示例/new_datas.txt'   # 数据集路径

# Vectorize the data.
input_texts = []  # 输入（源语言）序列
target_texts = []  # 输出（目标语言）序列
input_characters = set()   # 输入词典
target_characters = set()   # 目标词典

# --------------处理数据----------------------------------------------------------------
# -------------------------------------------------------------------------------------
with open(data_path, 'r', encoding='utf-8') as f:   # 读取数据集
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # 使用“tab”作为序列开始的特征，使用“\n”作为序列结束的特征
    target_text = '\t' + target_text + '\n'

    input_texts.append(input_text)
    target_texts.append(target_text)
    # 制作源语言的词典
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    # 制作目标语言的词典
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))      # 对源语言序列排序
target_characters = sorted(list(target_characters))    # 对目标语言序列排序
num_encoder_tokens = len(input_characters)             # 源语言词典大小
num_decoder_tokens = len(target_characters)            # 目标语言词典大小
max_encoder_seq_length = max([len(txt) for txt in input_texts])     # 输入序列的最大长度
max_decoder_seq_length = max([len(txt) for txt in target_texts])    # 输出序列的最大长度

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# mapping token to index， easily to vectors（建立字符到数字的映射）
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])        # 创建源语言字符index
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])      # 创建目标语言字符index

# 建立数字到字符的映射
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# np.zeros(shape, dtype, order)
# shape is an tuple, in here 3D
# 为了改变数据集的格式，decoder中有输入也有输出
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# input_texts contain all english sentences
# output_texts contain all chinese sentences
# zip('ABC','xyz') ==> Ax By Cz, looks like that
# the aim is: vectorilize text, 3D
# 将输入的中文或英文字符转换为one-hot形式
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        # 3D vector only z-index has char its value equals 1.0
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data不包含开始字符
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

def create_model():
    # 定义编码器的输入
    encoder_inputs = Input(shape=(None, num_encoder_tokens))  # None意味着可以输入不定长的序列
    # 返回状态
    encoder = LSTM(latent_dim, return_state=True)
    # 调用编码器，得到编码器的输出(输入其实不需要),以及状态信息 state_h 和 state_c
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    print("编码器的隐状态：", state_h)
    # 丢弃encoder_outputs, 保存state_h,state_c为了输入给decoder
    encoder_state = [state_h, state_c]

    # 定义解码器的输入
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # 并且返回其中间状态，中间状态在训练阶段不会用到，但是在推理阶段将是有用的
    decoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True)
    # 将编码器输出的状态作为初始解码器的初始状态
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_state)
    # 添加全连接层
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # 定义整个模型
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  # 模型的输入为encoder_inputs, decoder_inputs，输出为decoder_outputs

    # 定义 sampling 模型
    # 定义 encoder 模型，得到输出encoder_states
    encoder_model = Model(encoder_inputs, encoder_state)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

    # 得到解码器的输出以及中间状态
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)     # 将输入序列进行特征提取获得单元状态c和输出值h

    # this target_seq you can treat as initial state
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # 将这个空序列的内容设置为开始字符'\t'
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    # 进行字符恢复，简单起见，假设batch_size=1
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        # 以'\t'为开始，一个一个向后预测
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        print("预测下一个字符出现的概率：", output_tokens)   # 下一个字符出现的位置概率
        print("解码器隐状态：", h)
        print("上下文向量：", c)

        # Sample a token
        # argmax: Returns the indices of the maximum values along an axis
        # just like find the most possible char
        # 对下个字符采样  sampled_token_index是要预测下个字符最大概率出现在字典中的位置
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        print("下一个字符在字典中出现的位置：", sampled_token_index)
        # find char using index
        sampled_char = reverse_target_char_index[sampled_token_index]
        # and append sentence
        decoded_sentence += sampled_char
        # Exit condition: either hit max length
        # or find stop character.
        # 退出条件：生成 \n 或者 超过最大序列长度
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        # Update the target sequence (of length 1).
        # append then ?
        # creating another new target_seq
        # and this time assume sampled_token_index to 1.0
        # 更新target_seq
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        # Update states
        # update states, frome the front parts
        # 更新中间状态
        states_value = [h, c]
        print("中间状态：", states_value)
    return decoded_sentence

def train():
    model,encoder_model,decoder_model = create_model()
    #编译模型
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
    ##训练模型
    model.fit([encoder_input_data,decoder_input_data],decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)

    model.save('model/s2s.h5')
    encoder_model.save('model/encoder_model.h5')
    decoder_model.save('model/decoder_model.h5')


def test():
    encoder_model = load_model('model/encoder_model.h5', compile=False)
    decoder_model = load_model('model/decoder_model.h5', compile=False)
    ss = input("请输入要翻译的中文:")
    if ss == '-1':
        sys.exit()
    input_seq = np.zeros((1,max_encoder_seq_length,num_encoder_tokens))
    print("input_seq:",input_seq)
    for t, char in enumerate(ss):
        input_seq[0,t,input_token_index[char]]=1.0
    print("input_seq:",input_seq)
    decoded_sentence = decode_sequence(input_seq,encoder_model,decoder_model)
    print('-')
    print('Decoded sentence:', decoded_sentence)


if __name__ == '__main__':
    intro = input("select train model or test model:")
    if intro == "train":
        print("训练模式...........")
        train()
    else:
        print("测试模式.........")
        while (1):
            test()

