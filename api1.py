from question_classifier import *
from question_parser import *
from answer_search import *

class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    #cd C:/Code/KnowledgeGraph-QA-master
    #set
    #FLASK_APP=newchatbot.py
    def chat_main(self, sent):
        answer = '这个问题知识库中暂时没有，请查看用户问答库中有没有您要问的问题'
        res_classify = self.classifier.classify(sent)
        if not res_classify:
            return answer
        res_sql = self.parser.parser_main(res_classify)
        final_answers = self.searcher.search_main(res_sql)
        if not final_answers:
            return "这个问题知识库中暂时没有，请查看用户问答库中有没有您要问的问题。"
        else:
            return '\n'.join(final_answers)

from flask import Flask, request
from flask_cors import  CORS
app = Flask(__name__)
CORS(app, supports_credentials=True)
# CORS(app, resources=r'/*')


@app.route('/api', methods=['GET', 'POST'])
def indextest():
    handler = ChatBotGraph()
    if request.method == 'GET':
        question = request.args
        question = question['question']
        if question == '我想看看中国的疫情最新情况':
            answer = ''
            imgurl = './img/chinesemap.png'
            return {'answer': answer,'imgurl':imgurl, 'state':4}  # state=4 将返回的地址处理成图片
        if question == '我想看看世界各国的疫情最新情况':
            answer = ''
            imgurl = './img/world.png'
            return {'answer': answer,'imgurl':imgurl, 'state':4}

        else:
            answer = handler.chat_main(question)  # 接口1产生的answer（未return）
            imgurl = ''

            if answer == "这个问题知识库中暂时没有，请查看用户问答库中有没有您要问的问题。":
                return {'answer': answer,'imgurl':imgurl, 'state':1}   # 1代表跳转接口2
            else:   # 机器可以回答
                answer=answer+'\n可以解决你的问题吗？'
                imgurl = ''
                return {'answer': answer, 'imgurl':imgurl,'state':2}

    elif request.method == 'POST':
        question = request.form["question"]
        if '中国' and '疫情' in question:
            answer = ''
            imgurl = './img/chinese_map.png'
            return {'answer': answer,'imgurl':imgurl, 'state':4}  # state=4 将返回的地址处理成图片
        if '世界' and '确诊' in question:
            answer = ''
            imgurl = './img/world.png'
            return {'answer': answer,'imgurl':imgurl, 'state':4}
        else:   # 进行一问一答
            answer = handler.chat_main(question)  # 接口1产生的answer（未return）
            imgurl = ''
            if answer == "这个问题知识库中暂时没有，请查看用户问答库中有没有您要问的问题":
                return {'answer': answer,'imgurl':imgurl, 'state':2}   # 1代表跳转接口2
            else:   # 机器可以回答
                answer=answer+'\n可以解决你的问题吗？'
                imgurl = ''
                return {'answer': answer, 'imgurl':imgurl,'state':2}  # ！state=2第一次前端判断
                # 前端判断  用户输入可以或不可  可以输出感谢您的提问  不可以跳转接口2


@app.route('/', methods=['GET', 'POST'])
def a():
    return request.form['question']

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
#cd C:/Code/KnowledgeGraph-QA-master
