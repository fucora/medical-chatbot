from question_classifier import *
from question_parser import *
from answer_search import *
from sim_seq import *
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
'''问答类'''
class ChatBotGraph:
    def __init__(self):
        self.classifier = QuestionClassifier()
        self.parser = QuestionPaser()
        self.searcher = AnswerSearcher()

    def chat_main(self, sent):
        answer = '您好，我是爱医生智能助理，希望可以帮到您。如果没答上来，可联系我们xxxx@aidoctor.com。祝您身体棒棒！'
        res_classify = self.classifier.classify(sent)
        if not res_classify:
            return answer
        res_sql = self.parser.parser_main(res_classify)
        final_answers = self.searcher.search_main(res_sql)
        if not final_answers:
            return "这个问题知识库中暂时没有，请查看用户问答库中有没有您要问的问题。",
        else:
            return '\n'.join(final_answers)

    def xinli(self,question):
        result = client.sentimentClassify(question)
        data = result['items']
        items = data[0]
        positive_prob = items['positive_prob']
        negative_prob = items['negative_prob']
        sentiment = items['sentiment']
        if sentiment == 2 and positive_prob - negative_prob >= 0.05:
            return '祝您永远保持好心情！'
        if sentiment == 0 and negative_prob - positive_prob >= 0.05:
            return '很抱歉您这么难过，能和我谈谈您的心事吗'
        else:
            return '加油，奥利给！'



if __name__ == '__main__':
    handler = ChatBotGraph()
    sim = Word2vecSim()
    while 1:
        question = input('用户:')
        if question=='您好，我想进行心理咨询':
            print('请稍等，正在为您转接心理医生')
            question = input('用户:')
            answer=handler.xinli(question)
            print(answer)
        else:
            answer = handler.chat_main(question)
            if answer == "这个问题知识库中暂时没有，请查看用户问答库中有没有您要问的问题。":
                final_answer = sim.sim_main(question)
                print('爱医生智能助理:', final_answer)
            else:
                print('爱医生智能助理:',answer)
                print('可以解决你的问题吗？')
                x = input("请输入1表示可以，2表示不可以：")
                if x == '1':
                    print("感谢您的提问")
                elif x == '2':
                    print("很抱歉知识库中的知识不能回答您的问题，请查看用户问答库中有没有您要问的问题。")
                    final_answer = sim.sim_main(question)
                    print(final_answer)
                else:
                    print("输入出错请重新提问。")


