# coding:utf-8
import sys
import jieba


class WordToken(object):
    def __init__(self):
        # 最小起始id号, 保留的用于表示特殊标记
        self.START_ID = 4
        self.word2id_dict = {}
        self.id2word_dict = {}


    def load_file_list(self, file_list, min_freq):
        """
        加载样本文件列表，全部切词后统计词频，按词频由高到低排序后顺次编号
        并存到self.word2id_dict和self.id2word_dict中
        file_list = [question, answer]
        min_freq: 最小词频，超过最小词频的词才会存入词表
        """
        # 词频字典
        words_count = {}
        for file in file_list:
            with open(file, 'r', encoding='utf-8') as file_object:
                for line in file_object.readlines():
                    line = line.strip()
                    seg_list = jieba.cut(line)
                    for str in seg_list:
                        if str in words_count:
                            words_count[str] = words_count[str] + 1
                        else:
                            words_count[str] = 1

        sorted_list = [[v[1], v[0]] for v in words_count.items()]  # 解析字典
        sorted_list.sort(reverse=True)
        for index, item in enumerate(sorted_list):
            word = item[1]
            if item[0] < min_freq:     # 小于词频的不考虑
                break
            self.word2id_dict[word] = self.START_ID + index       # 使单词与其词频挂钩，进行排号
            self.id2word_dict[self.START_ID + index] = word
        return index

    def word2id(self, word):
        """判断word是不是字符串,从词到编号的转换"""
        if not isinstance(word, str):
            print("Exception: error word not unicode")
            sys.exit(1)         # 有错误退出
        if word in self.word2id_dict:
            return self.word2id_dict[word]
        else:
            return None

    def id2word(self, id):
        """编号转单词"""
        id = int(id)
        if id in self.id2word_dict:
            return self.id2word_dict[id]
        else:
            return None

