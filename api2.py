from aip import AipNlp
import numpy as np
APP_ID = '19349034'
API_KEY = 'Y5qY1MYmD2xV7LMeO5rFveXC'
SECRET_KEY = '2eMGn1PS1XofcsYam3WWkz9fBkyitg6w'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
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

a_type = list(zip(q_list,type_list))


from flask import Flask, request,render_template
from flask_cors import  CORS
app = Flask(__name__)
CORS(app, supports_credentials=True)


def sim_main(target):
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
        result3 = client.wordSimEmbedding(target_type, ty_list[i][1])
        word_sim = result3.get('score')
        if word_sim is None:
            word_sim = 0
        if word_sim >= 0.3:
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
    answer = a_list[q_list.index(res[0][0])]   # 返回相似度最高的问题对应的答案
    answer= answer+'您对这个答案满意吗'
    return answer


@app.route('/api', methods=['GET', 'POST'])
def indextest():
    if request.method == 'GET':
        question = request.args
        question = question['question']
        answer=sim_main(question)
        imgurl=''
        return{'answer':answer, 'imgurl':imgurl, 'state': 3}  # state3进行第二次前端判断 显示问题 您对这个结果满意吗

    elif request.method == 'POST':
        question = request.form["question"]
        answer=sim_main(question)
        imgurl=''
        return{'answer':answer, 'imgurl':imgurl,'state': 3}


@app.route('/', methods=['GET', 'POST'])
def a():
    return request.form['question']

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001,debug=True)