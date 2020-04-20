from selenium import webdriver
import os
from time import sleep
import xlwt

bookwork = xlwt.Workbook(encoding='utf-8')
sheetbook = bookwork.add_sheet('国内疫情数据')

title = ['地区','现有确诊','累计确诊','治愈','死亡']
for i in range(len(title)):
    sheetbook.write(0,i,title[i])
    # 限制图片加载
options = webdriver.ChromeOptions()
prefs = {
        'profile.default_content_setting_values': {
            'images': 2,   #限制图片加载
            'permissions.default.stylesheet': 2,    #限制css样式加载
            # 'javascript':2
        }
    }
options.add_experimental_option('prefs', prefs)
options.add_argument('headless')      #设置无浏览器界面运行
driver = webdriver.Chrome(chrome_options=options)

link = 'https://news.qq.com/zt2020/page/feiyan.htm#/?nojump=1'
driver.get(link)
sleep(3)
time = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[3]/div[1]/div[2]/p/span').text
print(time)
time = time.split(' ')
time = time[0]
print(time)
dir = driver.find_element_by_id('listWraper')

table = driver.find_elements_by_xpath('//*[@id="listWraper"]/table[2]/tbody')
print(len(table))

num = 1
for i in table:
    name = i.find_element_by_tag_name('th').text
    td = i.find_elements_by_tag_name('td')
    current = td[0].find_element_by_tag_name('p').text
    history = td[1].find_element_by_tag_name('p').text
    head = td[2].find_element_by_tag_name('p').text
    dead = td[3].find_element_by_tag_name('p').text
    sheetbook.write(num,0,name)
    sheetbook.write(num, 1,current )
    sheetbook.write(num, 2,history)
    sheetbook.write(num, 3,head)
    sheetbook.write(num, 4,dead)
    print(name,current,history,head,dead)
    num = num + 1
bookwork.save(time[5:]+'国内疫情数据.xlsx')
print('更新完成')
driver.quit()   #关闭浏览器