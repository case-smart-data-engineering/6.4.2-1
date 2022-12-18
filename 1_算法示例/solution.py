from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import numpy as np
import sys
import tensorflow.compat.v2 as tf

# 定义神经网络的参数
batch_size = 64   # 一次训练所选取的样本数
epochs = 500    # 训练轮数
latent_dim = 256  # LSTM的单元个数
num_samples = 3000   # 训练样本的大小
data_path = '1_算法示例/data.txt'   # 数据集路径

# 输入（源语言）序列，即输入的中文字符串
input_texts = []
# 输出（目标语言）序列，即对应的英文字符串
target_texts = []
# 输入词典，即用到的所有输入字符,如我，你，人....
input_characters = set()
# 目标词典，即用到的所有输出字符，如：a,b....
target_characters = set()

# --------------处理数据--------------------------------------------------------
# ------------------------------------------------------------------------------
# 读取数据集
with open(data_path, 'r', encoding='utf-8') as f:
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

# 对源语言序列排序
input_characters = sorted(list(input_characters))
# 对目标语言序列排序
target_characters = sorted(list(target_characters))
# 源语言词典大小(输入字符中不同中文字符的数量）
num_encoder_tokens = len(input_characters)
# 目标语言词典大小(输出字符中不同英文字符的数量）
num_decoder_tokens = len(target_characters)
# 输入序列的最大长度
max_encoder_seq_length = max([len(txt) for txt in input_texts])
# 输出序列的最大长度
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# 建立字符到数字的映射（字典），用于字符的向量化
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# 建立数字到字符的映射（字典），用于恢复
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

# 为了改变数据集的格式，decoder中有输入也有输出
# 需要把每条语料转换成LSTM需要的三维数据输入[n_samples, timestamp, one-hot feature]到模型中
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# zip('ABC','xyz') ==> Ax By Cz
# 将输入的中文或英文字符串进行one-hot编码
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data不包含开始字符tab
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
# ------------------------------------------------------------------------------

# --------------------建立模型---------------------------------------------------
def create_model():
    # 定义编码器的输入，None意味着可以输入一个不定长的序列
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    # 定义LSTM层，latent_dim为LSTM单元中每个门的神经元个数，return_state设为True时才会返回最后时刻的状态值h和c
    encoder = LSTM(latent_dim, return_state=True)
    # 调用编码器，得到编码器的输出，以及状态信息 state_h 和 state_c
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    print("编码器的隐状态：", state_h)
    # 丢弃encoder_outputs, 保存state_h,state_c为了输入给decoder
    encoder_state = [state_h, state_c]

    # 定义解码器的输入
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # return_sequences设为True表示会返回其中间状态，在推理输出字符的阶段将是有用的
    decoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True)
    # 将编码器输出的状态作为初始解码器的初始状态
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_state)
    # 添加全连接层，将输出全连接到num_decoder_tokens（所有的不重复的英文字符）维度上，
    # 激活函数选用'softmax',根据得到的每个字符的概率，判断decoder每一步的输出是什么
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # 定义整个模型，模型的输入为encoder_inputs, decoder_inputs，输出为decoder_outputs
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # 定义 sampling 模型（利用模型进行预测）
    # 定义 encoder 模型，得到输出encoder_states
    encoder_model = Model(encoder_inputs, encoder_state)

    # decoder从encoder中获得的要输入的单元状态信息h和c
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

    # 得到解码器的输出以及中间状态
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
    # 得到decoder中LSTM单元的输出状态信息h和c
    decoder_states = [state_h, state_c]
    # 添加全连接层
    decoder_outputs = decoder_dense(decoder_outputs)
    # 定义 decoder 模型
    decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model):
    # 将输入序列进行特征提取获得单元状态c和输出值h
    states_value = encoder_model.predict(input_seq)

    # 构建一个空序列
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # 将这个空序列的内容设置为开始字符'\t'
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        # 以'\t'为开头，结合状态向量states_value，开始一个一个向后预测
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # output_tokens是下一个字符出现的位置概率

        # 对下个字符采样  sampled_token_index是要预测下个字符最大概率出现在字典中的位置
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        print("下一个字符在字典中出现的位置：", sampled_token_index)
        # 使用数字到字母的映射，根据index找到字符
        sampled_char = reverse_target_char_index[sampled_token_index]
        # 将找到的字符加入到翻译结果decoded_sentence中
        decoded_sentence += sampled_char

        # 退出条件：生成 \n 或者 超过最大序列长度
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # 更新target_seq，将新预测得到的字符添加到target_seq中，作为下一轮预测的输入
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # 更新中间状态
        states_value = [h, c]
        # print("中间状态：", states_value)

    return decoded_sentence

# 训练模型
def train():
    model, encoder_model, decoder_model = create_model()
    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss='categorical_crossentropy')
    # 训练模型
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1)

    # 保存模型，方便测试使用
    model.save('model/s2s.h5')
    encoder_model.save('model/encoder_model.h5')
    decoder_model.save('model/decoder_model.h5')


def test():
    encoder_model = load_model('model/encoder_model.h5', compile=False)
    decoder_model = load_model('model/decoder_model.h5', compile=False)
    # 提示：因为数据集太小，训练的模型在做测试时效果不好，故在输入要翻译的中文时应选择数据集中给出的句子，例如：难以置信、我爱中国等
    # 若要翻译的是数据集外的句子，则会结束循环，程序运行结束
    ss = input("请输入要翻译的中文:")
    if ss == '-1':
        sys.exit()
    input_seq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens))
    for t, char in enumerate(ss):
        input_seq[0, t, input_token_index[char]]=1.0
    decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model)
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
