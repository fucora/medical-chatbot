# V1 版本开始支持链式调用
#coding = utf-8
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.globals import ThemeType  # 主题
from snapshot_selenium import snapshot as driver
from pyecharts.render import make_snapshot
import xlrd


filename = '04-15国内疫情数据.xlsx'
file = xlrd.open_workbook(filename)
sheet = file.sheet_by_name('国内疫情数据')
cityname = sheet.col_values(0)  # 获取城市名
number = sheet.col_values(1)  # 获取城市现有确诊人数
data = []
for i in range(1, len(cityname)):
    list = []
    list.append(cityname[i])
    list.append(number[i])
    data.append(list)

# 设置地图参数
map = (
    Map(init_opts=opts.InitOpts(bg_color="#FFFAFA", theme=ThemeType.ESSOS, width=1000))
        .add("现存确诊人数", data)
        .set_global_opts(
        title_opts=opts.TitleOpts(title=filename[0:5] + "国内数据的疫情图"),
        visualmap_opts=opts.VisualMapOpts(
            is_piecewise=True,  # 设置是否为分段显示
            # 自定义的每一段的范围，以及每一段的文字，以及每一段的特别的样式。例如：
            pieces=[
                {"min": 400, "label": '>400人', "color": "#eb2f06"},
                {"min": 300, "max": 400, "label": '300-400人', "color": "#FF3030"},  # 不指定 max，表示 max 为无限大（Infinity）。
                {"min": 150, "max": 300, "label": '150-300人', "color": "#FF4500"},
                {"min": 100, "max": 150, "label": '100-150人', "color": "#FF7F50"},
                {"min": 50, "max": 100, "label": '50-100人', "color": "#FFA500"},
                {"min": 1, "max": 50, "label": '1-50人', "color": "#FFDEAD"},
            ],
            # 两端的文本，如['High', 'Low']。
            range_text=['高', '低'],
        ),
    )
)
make_snapshot(driver, map.render('C:/Code/可视化疫情2/中国疫情人数地图.html'), "C:/Code/可视化疫情2/chinese_map.png")

