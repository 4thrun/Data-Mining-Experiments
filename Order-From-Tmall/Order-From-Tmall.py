#!/usr/bin/env python
# coding: utf-8

# # Order\-From\-Tmall
# 
# The three data sets come from [Heywhale](https://www.kesci.com/mw/dataset/5ffac64f3441fd001538228b/file) which are not related to each other\. And the project are divided into three parts accordingly\.

# In[1]:


# !pip install pyecharts==1.9.0
# !pip install pandas==1.3.0


# ## Part\-1: Tmall\_OrderReport\.csv
# 
# This set includes order information: time, province (in shipping address) as DIMENSIONS and sales volume, sales, refund amount, return rate, turnover rate, area, order time trend and so on as MEASURES\.

# ### 1\. Load Data 

# In[2]:


import pandas as pd 

# data = pd.read_csv('/home/aistudio/work/Tmall_OrderReport.csv')
data = pd.read_csv("./Tmall_OrderReport.csv")
data.head()


# In[3]:


data.info()


# In[4]:


data.columns = data.columns.str.strip() # remove space 
data.columns


# In[5]:


data[data.duplicated()].count() # no duplicated record ╰(￣ω￣ｏ)


# In[6]:


data.isnull().sum() # payment time null -> order not paid 


# In[7]:


data["收货地址"] = data["收货地址"].str.replace(pat='自治区|维吾尔|回族|壮族|省', repl='', regex=True) # simple data cleaning 
data["收货地址"].unique() # print 


# ### 2\. Visualization

# In[8]:


# overview 
result = {}
result['总订单数'] = data['订单编号'].count()  
result['已完成订单数'] = data['订单编号'][data['订单付款时间'].notnull()].count()  
result['未付款订单数'] = data['订单编号'][data['订单付款时间'].isnull()].count()  
result['退款订单数'] = data['订单编号'][data['退款金额'] > 0].count()  
result['总订单金额'] = data['总金额'][data['订单付款时间'].notnull()].sum()  
result['总退款金额'] = data['退款金额'][data['订单付款时间'].notnull()].sum()  
result['总实际收入金额'] = data['买家实际支付金额'][data['订单付款时间'].notnull()].sum()

result


# In[9]:


from pyecharts import options as opts 
from pyecharts.charts import Map, Bar, Line 
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts
from pyecharts.faker import Faker

table = Table()

headers = ['总订单数', '总订单金额', '已完成订单数', '总实际收入金额', '退款订单数', '总退款金额', '成交率', '退货率']
rows = [
    [
        result['总订单数'], f"{result['总订单金额']/10000:.2f} 万", result['已完成订单数'], f"{result['总实际收入金额']/10000:.2f} 万",
        result['退款订单数'], f"{result['总退款金额']/10000:.2f} 万", 
        f"{result['已完成订单数']/result['总订单数']:.2%}",
        f"{result['退款订单数']/result['已完成订单数']:.2%}",
    ]
]
table.add(headers, rows)
table.set_global_opts(
    title_opts=ComponentTitleOpts(title='整体情况')
)
table.render_notebook()


# In[10]:


# region analysis
result2 = data[data["订单付款时间"].notnull()].groupby("收货地址").agg({'订单编号': 'count'})
result2_1 = result2.to_dict()['订单编号']
c = (
    Map()
    .add("订单量", [*result2_1.items()], 'china', is_map_symbol_show=False)
    .set_series_opts(label_opts=opts.LabelOpts(is_show=True))
    .set_global_opts(
        title_opts=opts.TitleOpts(title='地区分布'),
        visualmap_opts=opts.VisualMapOpts(max_=1000),
    )
)
c.render_notebook()


# In[11]:


# daily trend 
data["订单创建时间"] = pd.to_datetime(data["订单创建时间"])
data["订单付款时间"] = pd.to_datetime(data["订单付款时间"])

result3_1 = data.groupby(data['订单创建时间'].apply(lambda x: x.strftime("%Y-%m-%d"))).agg({'订单编号': 'count'}).to_dict()['订单编号']
c =(
    Line()
    .add_xaxis(list(result3_1.keys()))
    .add_yaxis('订单量', list(result3_1.values()))
    .set_series_opts(
        label_opts = opts.LabelOpts(is_show=False),
        markpoint_opts=opts.MarkPointOpts(
            data = [
                opts.MarkPointItem(type_='max', name='最大值'),
                ]
        ),
    )
    .set_global_opts(title_opts=opts.TitleOpts(title='每日订单量趋势'))
)
c.render_notebook()


# Due to COVID-2019, sales was low in the first half of February\. But in the second half, sales increased significantly because of the resumption\.

# In[12]:


# hourly trend 
result3_2 = data.groupby(data['订单创建时间'].apply(lambda x: x.strftime("%H"))).agg({'订单编号': 'count'}).to_dict()['订单编号']
x = [*result3_2.keys()]
y = [*result3_2.values()]
c = (
    Bar()
    .add_xaxis(x)
    .add_yaxis('订单量', y)
    .set_global_opts(title_opts=opts.TitleOpts(title='每小时订单量趋势'))
    .set_series_opts(
        label_opts=opts.LabelOpts(is_show=False),
        markpoint_opts=opts.MarkPointOpts(
            data=[
                opts.MarkPointItem(type_='max', name='峰值'),
                opts.MarkPointItem(name='第二峰值', coord=[x[15], y[15]], value=y[15]),
                opts.MarkPointItem(name='第三峰值', coord=[x[10], y[10]], value=y[10]),
            ]
        ),
    )
)
c.render_notebook()


# The three peak periods: 10, 15, 21 make it clear for retailers how to arrange customer service, especially during 21:00 to 22:00\.

# In[13]:


# average time cost 
s = data['订单付款时间'] - data['订单创建时间']
s[s.notnull()].apply(lambda x : x.seconds/60).mean()


# ## Part\-2: Tmall_Double11Makeup\.csv
# 
# Valuable: date, store name as DIMENSIONS and sales, comments and so on as MEASURES\.

# ### 1\. Load Data 

# In[14]:


import pandas as pd 

# data2 = pd.read_csv('/home/aistudio/work/Tmall_Double11Makeup.csv')
data2 = pd.read_csv('./Tmall_Double11Makeup.csv')
data2.head()


# In[15]:


data2.info()


# In[16]:


data2[data2.duplicated()].count() # 86 completely duplicate records


# In[17]:


data2.drop_duplicates(inplace=True)
data2.reset_index(drop=True, inplace=True)
data2.isnull().sum()


# In[18]:


data2.fillna(0, inplace=True)
data2['update_time'] = pd.to_datetime(data2['update_time']).apply(lambda x: x.strftime("%Y-%m-%d"))
data2[data2['sale_count']>0].sort_values(by=["sale_count"]).head()


# In[19]:


# Add a column to indicate sales
data2['sale_amount'] = data2['price'] * data2['sale_count']
data2[data2['sale_count']>0].sort_values(by=['sale_count'])


# ### 2\. Visualization

# In[20]:


# daily overall sales trend
result = data2.groupby('update_time').agg({'sale_count': 'sum'}).to_dict()['sale_count']
c = (
    Line()
    .add_xaxis(list(result.keys()))
    .add_yaxis('销售量', list(result.values()))
    .set_series_opts(
        areastyle_opts = opts.AreaStyleOpts(opacity=0.5),
        label_opts = opts.LabelOpts(is_show=False), 
        markpoint_opts = opts.MarkPointOpts(
            data=[
                opts.MarkPointItem(type_='max', name='最大值'),
                opts.MarkPointItem(type_='min', name='最小值'),
                opts.MarkPointItem(type_='average', name='平均值'), 
            ]
        ),
    )
    .set_global_opts(title_opts=opts.TitleOpts(title='每日整体销售量趋势'))
)
c.render_notebook()


# In[21]:


# see which sells the best 
dts = list(data2['update_time'].unique())
dts.reverse()
dts 


# In[22]:


# draw 
from pyecharts import options as opts 
from pyecharts.charts import Map, Timeline, Bar, Line, Pie 
from pyecharts.components import Table 
from pyecharts.options import ComponentTitleOpts

tl = Timeline()
tl.add_schema(
    is_auto_play=False, 
    is_loop_play=True , 
    play_interval=500,
)
for dt in dts: 
    item = data2[data2['update_time']<=dt]     .groupby("店名").agg({'sale_count': 'sum', 'sale_amount': 'sum'})     .sort_values(by='sale_count', ascending=False)[:10]     .to_dict()
    bar = (
        Bar()
        .add_xaxis([*item['sale_count'].keys()])
        .add_yaxis("销售量", [round(val/10000,2) for val in item['sale_count'].values()], label_opts=opts.LabelOpts(position="right", formatter='{@[1]/} 万'))
        .add_yaxis("销售额", [round(val/10000/10000,2) for val in item['sale_amount'].values()], label_opts=opts.LabelOpts(position="right", formatter='{@[1]/} 亿元'))
        .reversal_axis()
        .set_global_opts(
            title_opts=opts.TitleOpts("累计销售量排行 TOP10")
        )
    )
    tl.add(bar, dt)
tl.render_notebook()


# In[23]:


item = data2.groupby('店名').agg({'sale_count': 'sum'}).sort_values(by='sale_count', ascending=False)[:10].to_dict()['sale_count']
item = {k: round(v/10000, 2) for k, v in item.items()}
c = (
    Pie()
    .add('销量', [*item.items()])
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c} 万（{d}%）"))
)
c.render_notebook()


# In[24]:


# see which is the most expensive 
item = data2.groupby('店名').agg({'price': 'mean'}).sort_values(by='price', ascending=False)[:20].sort_values(by="price").to_dict()
c = (
    Bar()
    .add_xaxis([*item['price'].keys()])
    .add_yaxis('销售量', [round(v, 2) for v in item['price'].values()], label_opts=opts.LabelOpts(position="right"),)
    .reversal_axis()
    .set_global_opts(title_opts=opts.TitleOpts(title='平均价格排行 Top20 '))
)
c.render_notebook()


# ## Part\-3: Tmall.DailyChemical\.xlsx
# 
# Wholesale orders including an order sheet and a commodity sheet\. Valuable: date, region, commodity as DIMENSIONS and sales, growing rate and so on as MEASURES\.

# ### 1\. Load Data 

# In[25]:


import pandas as pd 

# fact_order = pd.read_excel('/home/aistudio/work/Tmall_DailyChemical.xlsx', sheet_name='销售订单表')
# dim_product = pd.read_excel('/home/aistudio/work/Tmall_DailyChemical.xlsx', sheet_name='商品信息表')
fact_order = pd.read_excel('./Tmall_DailyChemical.xlsx', sheet_name='销售订单表')
dim_product = pd.read_excel('./Tmall_DailyChemical.xlsx', sheet_name='商品信息表')


# In[26]:


# simple data cleaning 
dim_product.head()


# In[27]:


dim_product.describe()


# In[28]:


dim_product[dim_product.duplicated()].count() # no duplicated records 


# In[29]:


dim_product[dim_product['商品编号'].duplicated()].count() # no duplicated ID


# In[30]:


dim_product.isnull().sum() # no null 


# In[31]:


# simple data cleaning 
fact_order.head()


# In[32]:


fact_order.info()


# In[33]:


fact_order[fact_order.duplicated()].count() # a few duplicated records 


# In[34]:


# delete the duplicated 
fact_order.drop_duplicates(inplace=True)
fact_order.reset_index(drop=True, inplace=True)
fact_order.isnull().sum() # a few null 


# In[35]:


# fill in null
fact_order.fillna(method='bfill', inplace=True)
fact_order.fillna(method='ffill', inplace=True)
fact_order.isnull().sum() 


# In[36]:


fact_order['订单日期'] = fact_order['订单日期'].apply(lambda x: pd.to_datetime(x, format='%Y#%m#%d') if isinstance(x, str) else x)
fact_order[fact_order['订单日期'] > '2021-01-01'] # one piece of dirty record (2050！)


# In[37]:


# filter the dirty
fact_order = fact_order[fact_order['订单日期'] < '2021-01-01']
fact_order['订单日期'].max(), fact_order["订单日期"].min()


# In[38]:


# data type conversion
fact_order['订购数量'] = fact_order['订购数量'].apply(lambda x: x.strip('个') if isinstance(x, str) else x).astype('int')
fact_order['订购单价'] = fact_order["订购单价"].apply(lambda x: x.strip("元") if isinstance(x, str) else x).astype('float')
fact_order["金额"] = fact_order['金额'].astype('float')
fact_order.info()


# In[39]:


# province 
fact_order['所在省份'] = fact_order['所在省份'].str.replace(pat='自治区|维吾尔|回族|壮族|省|市', repl='', regex=True) 
fact_order['所在省份'].unique()


# In[40]:


fact_order['客户编码'] = fact_order['客户编码'].str.replace(pat='编号', repl='', regex=False)
fact_order['客户编码'].head()


# ### 2\. Visualization 

# In[41]:


from pyecharts import options as opts 
from pyecharts.charts import Map, Bar, Line 
from pyecharts.components import Table 
from pyecharts.options import ComponentTitleOpts
from pyecharts.faker import Faker # only for test

# monthly ordering 
fact_order['订单月份'] = fact_order['订单日期'].apply(lambda x: x.month)
items = fact_order.groupby("订单月份").agg({'订购数量': 'sum', '金额': 'sum'}).to_dict()
x = [f'{key} 月' for key in items['订购数量'].keys()]
y_1 = [round(val/10000, 2) for val in items['订购数量'].values()]
y_2 = [round(val/10000/10000, 2) for val in items['金额'].values()]
c = (
    Bar()
    .add_xaxis(x)
    .add_yaxis('订购数量（万件）', y_1, is_selected=False)
    .add_yaxis('金额（亿元）', y_2)
    .set_global_opts(title_opts=opts.TitleOpts(title='每月订购情况'))
    .set_series_opts(label_opts=opts.LabelOpts(is_show=True),)
) 
c.render_notebook()


# In[42]:


# where do people enjoying making up most 
item = fact_order.groupby('所在地市') .agg({'订购数量': 'sum'}) .sort_values(by='订购数量', ascending=False)[:20] .sort_values(by='订购数量').to_dict()['订购数量']

c = (
    Bar()
    .add_xaxis([*item.keys()])
    .add_yaxis(
        '订购量', [round(v/10000, 2) for v in item.values()], 
        label_opts=opts.LabelOpts(position='right', formatter='{@[1]/} 万')
        )
    .reversal_axis()
    .set_global_opts(title_opts=opts.TitleOpts(title='订购数量排行 Top20'))
)
c.render_notebook()


# In[43]:


# which is in the greatest demand 
order = pd.merge(fact_order, dim_product, on='商品编号', how='inner') # table association
order 


# In[44]:


order.groupby(['商品大类', '商品小类']).agg({'订购数量': 'sum'}).sort_values(by=['商品大类', '订购数量'], ascending=[True, False])


# In[45]:


# which provinces have the greatest demand for cosmetics
item = fact_order.groupby("所在省份").agg({'订购数量': 'sum'}).to_dict()['订购数量']
c = (
    Map()
    .add('订购数量', [*item.items()], 'china', is_map_symbol_show=False)
    .set_series_opts(
        label_opts=opts.LabelOpts(is_show=True),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title='省份分布'),
        visualmap_opts=opts.VisualMapOpts(max_=1000000), 
    )
)
c.render_notebook()


# ### 3\. Customer Value 
# 
# Try to reveal customer value with the help of RFM model\.
# 
# RFM:
# - R\-Recency, last purchase time 
# - F\-Frequency, comsumption frequency
# - M\-Monetary, consumption amount 
# 
# A weight has to be set beforehand, for example, R\-20% F\-30% M\-50%, based on which a score can be given to measure customer value\.

# In[46]:


data_rfm = fact_order.groupby('客户编码').agg({"订单日期": 'max', '订单编码': 'count', '金额': 'sum'})
data_rfm.columns = ['最近一次购买时间', '消费频率', '消费金额']

data_rfm['R'] = data_rfm['最近一次购买时间'].rank(pct=True) #pct: percentage
data_rfm['F'] = data_rfm['消费频率'].rank(pct=True)
data_rfm['M'] = data_rfm['消费金额'].rank(pct=True)
data_rfm.sort_values(by='R', ascending=False)


# In[47]:


data_rfm['score'] = data_rfm['R']*20 + data_rfm['F']*30 + data_rfm["M"]*50
data_rfm['score'] = data_rfm['score'].round(1)
data_rfm.sort_values(by="score", ascending=False)

