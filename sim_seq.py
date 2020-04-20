import numpy as np
from aip import AipNlp
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops  import seq2seq
import word_token
import jieba
import random
import bm25_fitness_data
size = 8               # LSTM神经元size
GO_ID = 1              # 输出序列起始标记
EOS_ID = 2             # 结尾标记
PAD_ID = 0             # 空值填充0
min_freq = 1           # 样本频率超过这个值才会存入词表
epochs = 20000          # 训练次数
batch_num = 1000       # 参与训练的问答对个数
input_seq_len = 25         # 输入序列长度
output_seq_len = 50        # 输出序列长度
init_learning_rate = 0.5     # 初始学习率
wordToken = word_token.WordToken()   # 这是个词袋模型

# 放在全局的位置，为了动态算出 num_encoder_symbols 和 num_decoder_symbols
max_token_id = wordToken.load_file_list(['./samples/question', './samples/answer'], min_freq)
num_encoder_symbols = max_token_id + 5    # 算上加上填充、结尾标记、输出标记
num_decoder_symbols = max_token_id + 5
APP_ID = '19349034'
API_KEY = 'Y5qY1MYmD2xV7LMeO5rFveXC'
SECRET_KEY = '2eMGn1PS1XofcsYam3WWkz9fBkyitg6w'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)  # 调用百度词,句相似度API
options = {}
options["model"] = "bert"


q_list=[] # question
a_list=[] # answer
type_list=[] #ill type
with open('C:/Code/KnowledgeGraph-QA-master/data/question.txt', encoding="utf-8") as f:
    for line in f.readlines():
        q_list.append(line.strip())
with open('C:/Code/KnowledgeGraph-QA-master/data/answer.txt', encoding="utf-8") as f:
    for line in f.readlines():
        a_list.append(line.strip())
with open('C:/Code/KnowledgeGraph-QA-master/data/ill_type.txt', encoding="utf-8") as f:
    for line in f.readlines():
        type_list.append(line.strip())
ty_list, s, p = np.unique(type_list, return_index=True, return_inverse=True)

a_type = list(zip(q_list,type_list))   # [[answer],[type]]  如[['老是睡不着'],['失眠症']]



class Word2vecSim:

    def sim_main(self, target):
        res = []
        target_type = []
        result2 = client.lexer(target)
        result2 = result2.get('items')
        if result2 is None:
            pass
        else:
            for i in range(len(result2)):
                type = result2[i]
                item = type.get('item')
                pos = type.get('pos')
                if pos == 'nz':
                    target_type = item           # 利用词性分析提取出专有名词，绝大部分情况为疾病名称

        for i in range(len(ty_list)):
            result3=client.wordSimEmbedding(target_type, ty_list[i][1])
            word_sim=result3.get('score')
            if word_sim is None:
                word_sim=0
            if word_sim>=0.3:     # 看问句属于哪种type，然后再计算相似度，减少运算消耗
                for string in q_list:
                    result = client.simnet(string, target, options)
                    score = result.get('score')
                    if score is None:
                        score = 0
                    res.append([string, score])
            else:
                for string in q_list:
                    score = 0
                    res.append([string, score])

        res = sorted(res, key=lambda x: x[1], reverse=True)
        for i in range(5):
            print(i + 1, res[i][0])
        print(6, "以上都不是,生成答案")
        x = input("请输入你要问的问题序号：")
        if int(x) == 6:
            answer=seq_predict(target)
            return answer

        elif 0 < int(x) < 6:
            return a_list[q_list.index(res[int(x) - 1][0])]
        else:
            return "无效输入，请重新提问"


    '''def sim_eval(self, target):
        res = []
        target_type = []
        result2 = client.lexer(target)
        result2 = result2.get('items')
        if result2 is None:
            pass
        else:
            for i in range(len(result2)):
                type = result2[i]
                item = type.get('item')
                pos = type.get('pos')
                if pos == 'nz':
                    target_type = item

        for i in range(len(ty_list)):
            result3=client.wordSimEmbedding(target_type, ty_list[i][1])
            word_sim=result3.get('score')
            if word_sim is None:
                word_sim=0
            if word_sim>=0.3:
                for string in q_list:
                    result = client.simnet(string, target, options)
                    score = result.get('score')
                    if score is None:
                        score = 0
                    res.append([string, score])

            else:
                for string in q_list:
                    score = 0
                    res.append([string, score])

        res = sorted(res, key=lambda x: x[1], reverse=True)
        candidate = ''
        for i in range(5):
            candidate = candidate + str(i + 1) + res[i][0] + '\n'
        return candidate'''


#seq模型部分
def get_model(feed_previous=False):
    learning_rate = tf.Variable(float(init_learning_rate), trainable=False, dtype=tf.float32)
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)

    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    for i in range(input_seq_len):
        encoder_inputs.append(tf.compat.v1.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
    for i in range(output_seq_len + 1):
        decoder_inputs.append(tf.compat.v1.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
    for i in range(output_seq_len):
        target_weights.append(tf.compat.v1.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

    # decoder_inputs左移一个时序作为targets
    targets = [decoder_inputs[i + 1] for i in range(output_seq_len)]

    cell = tf.contrib.rnn.BasicLSTMCell(size)

    # 这里输出的状态我们不需要
    outputs, _ = seq2seq.embedding_attention_seq2seq(
        encoder_inputs,
        decoder_inputs[:output_seq_len],
        cell,
        num_encoder_symbols=num_encoder_symbols,
        num_decoder_symbols=num_decoder_symbols,
        embedding_size=size,
        output_projection=None,
        # 是一个(W, B)结构的tuple，W是shape为[output_size x num_decoder_symbols]的weight矩阵，B是shape为[num_decoder_symbols]的偏置向量
        feed_previous=feed_previous,
        dtype=tf.float32)

    # 计算交叉熵损失
    loss = seq2seq.sequence_loss(outputs, targets, target_weights)
    # 梯度下降优化器
    opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    # 优化目标：让loss最小化
    update = opt.apply_gradients(opt.compute_gradients(loss))
    # 模型持久化,保存所有的变量
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    return encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate


def get_id_list_from(sentence):
    """
    得到分词后的ID
    """
    sentence_id_list = []
    seg_list = jieba.cut(sentence)
    for str in seg_list:
        id = wordToken.word2id(str)
        if id:
            sentence_id_list.append(wordToken.word2id(str))
    return sentence_id_list


def get_samples(train_set, batch_num):
    """
    构造样本数据:传入的train_set是处理好的问答集
    batch_num:让train_set训练集里多少问答对参与训练
    encoder_inputs： []
    """
    raw_encoder_input = []
    raw_decoder_input = []
    if batch_num >= len(train_set):
        batch_train_set = train_set
    else:
        random_start = random.randint(0, len(train_set) - batch_num)
        batch_train_set = train_set[random_start:random_start + batch_num]

    # 添加起始标记、结束填充
    for sample in batch_train_set:
        raw_encoder_input.append([PAD_ID] * (input_seq_len - len(sample[0])) + sample[0])
        raw_decoder_input.append([GO_ID] + sample[1] + [PAD_ID] * (output_seq_len - len(sample[1]) - 1))

    encoder_inputs = []
    decoder_inputs = []
    target_weights = []

    for length_idx in range(input_seq_len):
        encoder_inputs.append(np.array([encoder_input[length_idx] for encoder_input in raw_encoder_input],
                                       dtype=np.int32))  # dtype是RNN状态数据的类型，默认是tf.float32
    for length_idx in range(output_seq_len):
        decoder_inputs.append(
            np.array([decoder_input[length_idx] for decoder_input in raw_decoder_input], dtype=np.int32))
        target_weights.append(np.array([
            0.0 if length_idx == output_seq_len - 1 or decoder_input[length_idx] == PAD_ID else 1.0 for decoder_input in
            raw_decoder_input
        ], dtype=np.float32))
    return encoder_inputs, decoder_inputs, target_weights


def seq_to_encoder(input_seq):
    """
    从输入空格分隔的数字id串，转成预测用的encoder、decoder、target_weight等
    """
    input_seq_array = [int(v) for v in input_seq.split()]
    encoder_input = [PAD_ID] * (input_seq_len - len(input_seq_array)) + input_seq_array
    decoder_input = [GO_ID] + [PAD_ID] * (output_seq_len - 1)
    encoder_inputs = [np.array([v], dtype=np.int32) for v in encoder_input]
    decoder_inputs = [np.array([v], dtype=np.int32) for v in decoder_input]
    target_weights = [np.array([1.0], dtype=np.float32)] * output_seq_len
    return encoder_inputs, decoder_inputs, target_weights


def seq_predict(question):
    """
    预测过程
    """
    with tf.compat.v1.Session() as sess:
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = get_model(
            feed_previous=True)

        input_seq = question
        max_score, answer = bm25_fitness_data.get_fitness_answer(input_seq)
        if max_score > 0.08:
            print("爱医生智能助理：" + str(answer))
        else:
            saver.restore(sess, 'C:/Code/KnowledgeGraph-QA-master/seq2seq/model/')
            input_seq = input_seq.strip()
            input_id_list = get_id_list_from(input_seq)  # 分词，得到id序列
            if len(input_id_list):
                sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = seq_to_encoder(
                    ' '.join([str(v) for v in input_id_list]))
                input_feed = {}
                for l in range(input_seq_len):
                    input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
                for l in range(output_seq_len):
                    input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                    input_feed[target_weights[l].name] = sample_target_weights[l]
                # GO_ID需要去了
                input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)

                # 预测输出
                outputs_seq = sess.run(outputs, input_feed)
                # 因为输出数据每一个是num_decoder_symbols维的，因此找到数值最大的那个就是预测的id，就是这里的argmax函数的功能
                outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in outputs_seq]
                outputs_seq.remove(PAD_ID)
                # 如果是结尾符，那么后面的语句就不输出了
                if EOS_ID in outputs_seq:
                    outputs_seq = outputs_seq[: outputs_seq.index(EOS_ID)]
                outputs_seq = [wordToken.id2word(v) for v in outputs_seq]
                print("爱医生智能助理", "".join(outputs_seq))
            else:
                print("：我好像不明白你在说什么")

