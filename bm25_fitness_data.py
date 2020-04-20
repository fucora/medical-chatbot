# coding = utf-8

import jieba
import re
import get_stopwords
from gensim.summarization import bm25


def punctuation_delete(line):
    """删除标点符号"""
    string = re.sub(" [+_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）,.?!]", "",line)
    return string


def stopwords_delete(line):
    """删除停用词"""
    stopwords = {}.fromkeys(get_stopwords.get_stopwords_list())
    final_question = []
    for word in line:
        if word not in stopwords:
            final_question.append(word)
    return final_question


def get_final_input(input_seq):
    """预处理输入的话"""
    input_seq.strip()
    input_seq = punctuation_delete(input_seq)
    input_seq = jieba.cut(input_seq, cut_all=True)
    final_input_seq = stopwords_delete(input_seq)
    return final_input_seq


def get_bm_data():
    """得到问题列表和问答对列表"""
    question_list = []
    question_answer_dict = {}
    with open('./data/question.txt', 'r', encoding='utf-8') as question_file:
        with open('./data/answer.txt', 'r', encoding='utf-8') as answer_file:
            while True:
                question = question_file.readline()
                answer = answer_file.readline()
                if question and answer:
                    question = question.strip()
                    answer = answer.strip()
                    question = punctuation_delete(question)
                    question = jieba.cut(question, cut_all=True)
                    final_question = stopwords_delete(question)
                    question_list.append(final_question)
                    question_answer_dict[answer] = final_question
                else:
                    break
    question_file.close()
    answer_file.close()
    return question_list, question_answer_dict


def get_key(dict, value):
    """从value索引到key"""
    return [k for k, v in dict.items() if v == value]


def get_fitness_answer(input_seq):
    """最后用bm25解决问题"""
    input_seq = get_final_input(input_seq)
    question_list, question_answer_direct = get_bm_data()
    bm25Model = bm25.BM25(question_list)
    scores = bm25Model.get_scores(input_seq)
    max_score = max(scores)
    idx = scores.index(max(scores))
    answer = get_key(question_answer_direct, question_list[idx])
    answer = str(answer[0])
    return max_score, answer

#     print(answer)
#
# lll = "如何练肩"
# get_fitness_answer(lll)




