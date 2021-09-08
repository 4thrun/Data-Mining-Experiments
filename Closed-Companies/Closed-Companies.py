#!/usr/bin/env python
# coding: utf-8

# # Closed\-Companies
# 
# The dataset was collected from [Heywhale](https://www.kesci.com/mw/dataset/5e023cd12823a10036af49b4/file): 6,272 records, 21 fields\.

# ## 1\. Load Data 

# In[ ]:


import pandas as pd 

data = pd.read_csv('./closed.csv')
data.head()


# ## 2\. Visualization
# 
# 

# In[ ]:


# region 
from pyecharts import options as opts 
from pyecharts.charts import Map

data['com_addr'] = data['com_addr'].apply(lambda x: x.strip())
s=data.groupby('com_addr').size()
c = (
    Map()
    .add('倒闭企业数量', [*s.items()], 'china')
    .set_global_opts(
        title_opts = opts.TitleOpts(title='地区分布'),
        visualmap_opts = opts.VisualMapOpts(max_=200),
    )
)
c.render_notebook()


# In[ ]:


# top 10
from pyecharts import options as opts 
from pyecharts.charts import Bar 
from pyecharts.faker import Faker 

s = data.groupby("cat").size().sort_values(ascending=False)[:10].to_dict()

c = (
    Bar()
    .add_xaxis(list(s.keys()))
    .add_yaxis('倒闭企业数量', list(s.values()))
    .set_global_opts(title_opts=opts.TitleOpts(title='行业排行TOP10'))
)
c.render_notebook()


# In[ ]:


# subdivide的
s = data.groupby('se_cat').size().sort_values(ascending=False)[:20].sort_values(ascending=True).to_dict()

c = (
    Bar()
    .add_xaxis(list(s.keys()))
    .add_yaxis('倒闭企业数量', list(s.values()))
    .reversal_axis()
    .set_series_opts(label_opts=opts.LabelOpts(position='right'))
    .set_global_opts(title_opts=opts.TitleOpts(title='细分领域TOP20'))
)
c.render_notebook()


# In[ ]:


# group by year 
data['born_year'] = data['born_data'].apply(lambda x: x[:4])
data['death_year'] = data['death_data'].apply(lambda x: x[:4])
s1 = data.groupby("born_year").size()
s2 = data.groupby('death_year').size()
s1 = pd.DataFrame({'year': s1.index, 'born': s1.values})
s2 = pd.DataFrame({'year': s2.index, "death": s2.values})
s = pd.merge(s1, s2, on='year', suffixes=['born', 'death'])
s = s[s['year']>'2008']

c = (
    Bar()
    .add_xaxis(s["year"].to_list())
    .add_yaxis('新生企业数量', s["born"].to_list())
    .add_yaxis('倒闭企业数量', s['death'].to_list())
    .set_global_opts(title_opts=opts.TitleOpts(title='年份分布'))
)
c.render_notebook()


# In[ ]:


# survival time 
def survival(x):
    if x< 365:
        return '少于1年'
    elif x < 365 * 2:
        return '1~2年'
    elif x < 365 * 3:
        return '2~3年'
    elif x < 365 * 4:
        return '3~4年'
    elif x < 365 * 5:
        return '4~5年'
    elif x < 365 * 10:
        return '5~10年'
    else:
        return '10年以上'

s = data.groupby(data['live_days'].apply(lambda x: survival(x))).size()

from pyecharts.charts import Pie
from pyecharts import options as opts 

p = (
    Pie()
    .add('', [*s.items()])
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    .set_global_opts(title_opts=opts.TitleOpts(title="企业存活时间"))
)
p.render_notebook()


# In[ ]:


# word cloud of investors
from pyecharts import options as opts 
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType

investor = dict()
for row in data["invest_name"].values:
    if not pd.isnull(row):
        for name in row.split('&'):
            investor[name] = investor.get(name, 0) + 1
investor = [*investor.items()]
investor.sort(key=lambda x: x[1], reverse=True)
c = (
    WordCloud()
    .add("", investor[:150], word_size_range=[20, 100], shape=SymbolType.DIAMOND)
    .set_global_opts(title_opts=opts.TitleOpts('投资人词云'))
)
c.render_notebook()


# In[ ]:


# word cloud of reasons 
reasons = dict()
for row in data['death_reason'].values:
    if not pd.isnull(row):
        for name in row.split(" "):
            reasons[name] = reasons.get(name, 0) + 1
c = (
    WordCloud()
    .add("", [*reasons.items()], word_size_range=[20, 100], shape=SymbolType.DIAMOND)
    .set_global_opts(title_opts=opts.TitleOpts(title="倒闭原因词云"))
)
c.render_notebook()


# In[ ]:


# word cloud of CEO
import jieba 

ceo_des = dict()
for row in data["ceo_per_des"].values:
    if not pd.isnull(row):
        result = jieba.lcut(row)
        for name in result:
            if len(name) == 1:
                break 
            ceo_des[name] = ceo_des.get(name, 0) + 1
ceo_des = [*ceo_des.items()]
ceo_des.sort(key=lambda x: x[1], reverse=True)
c = (
    WordCloud()
    .add("", ceo_des[:100], word_size_range=[20, 100], shape=SymbolType.DIAMOND)
    .set_global_opts(title_opts=opts.TitleOpts(title="CEO描述词云"))
)
c.render_notebook()

