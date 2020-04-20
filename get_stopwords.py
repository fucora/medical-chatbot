# -*- coding:utf-8 -*-

def get_stopwords_list():
    f = open(r'./stopwords','r', encoding='utf-8')
    a = f.readline()
    a = [i.strip() for i in f]
    f.close()
    return a
