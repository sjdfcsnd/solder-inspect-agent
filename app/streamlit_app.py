# -*- coding: utf-8 -*-
"""复核界面：上传一张板面照片，得到缺陷清单，并可对标记为「待复核」的焊点改判。

启动：streamlit run app/streamlit_app.py
"""

import json
import os
import sys

import cv2
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.agent import SolderAgent  # noqa: E402

st.set_page_config(page_title='焊点质量视觉复核 Agent', layout='wide')
st.title('焊点质量视觉复核 Agent')
st.caption('电子实训场景：拍一张板面照片，输出缺陷清单与置信度，低置信度与高影响缺陷转人工确认。')

with st.sidebar:
    st.header('运行设置')
    classifier = st.selectbox('分类器', ['resnet18', 'shallow', 'svm'], index=0)
    locator = st.selectbox('定位路径', ['auto', 'yolo', 'classic'], index=0)
    th = st.slider('置信度阈值', 0.20, 0.60, 0.35, 0.01)
    expected = st.number_input('板面设计焊点数', 1, 200, 24)
    st.divider()
    st.caption('阈值越低越不容易漏检，但进入复核队列的焊点越多。')

uploaded = st.file_uploader('上传板面照片（建议 1280×720 以上的正射图）', type=['png', 'jpg', 'jpeg'])

if uploaded is not None:
    tmp = os.path.join(ROOT, 'results', '_upload' + os.path.splitext(uploaded.name)[1])
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(tmp, 'wb') as f:
        f.write(uploaded.getbuffer())

    with st.spinner('运行中：门控 → 定位 → 分类 → 路由'):
        agent = SolderAgent(locator=locator, classifier=classifier, th=th,
                            expected_joints=expected)
        res = agent.run(tmp, save_dir=os.path.join(ROOT, 'results'))

    c1, c2, c3, c4 = st.columns(4)
    s = res['summary']
    c1.metric('焊点数', s['n_joints'])
    c2.metric('判定缺陷', s['n_defect'])
    c3.metric('待复核', s['n_review'])
    c4.metric('单图耗时', '%.2f s' % res['elapsed_s'])

    if res['status'] != 'ok':
        st.error(res['message'])
    else:
        st.success(res['message'])

    left, right = st.columns([1.25, 1])
    with left:
        if res.get('annotated') and os.path.exists(res['annotated']):
            st.image(res['annotated'], caption='绿=自动通过且正常，红=自动通过且判为缺陷，黄=待人工复核',
                     use_container_width=True)
    with right:
        if res['joints']:
            df = pd.DataFrame([{'序号': j['index'], '类别': j['label'], '置信度': j['conf'],
                                '处置': '复核' if j['action'] == 'review' else '自动',
                                '依据': j['reason']} for j in res['joints']])
            st.dataframe(df, height=420, use_container_width=True)

    if res['joints']:
        st.subheader('人工复核')
        st.caption('选择需要改判的焊点序号与结论，提交后写入复核记录（可回流为下一轮训练样本）。')
        idx = st.selectbox('焊点序号', [j['index'] for j in res['joints']
                                      if j['action'] == 'review'] or [1])
        new_label = st.selectbox('改判为', ['正常', '桥连', '虚焊少锡', '多锡', '偏移'])
        if st.button('提交复核'):
            rec = {'image': res['image'], 'index': idx, 'new_label': new_label,
                   'threshold': th, 'classifier': classifier}
            path = os.path.join(ROOT, 'results', 'review_log.jsonl')
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            st.success('已记录改判：第 %d 号 → %s。记录已写入 results/review_log.jsonl。'
                       % (idx, new_label))
            st.info('改判样本可用于下一轮微调，这是把人工知识回流到模型的入口。')

    with st.expander('查看完整 JSON 结果'):
        st.code(json.dumps(res, ensure_ascii=False, indent=2), language='json')
else:
    st.info('等待上传图片。可先用 tests/samples 下的样例图测试。')
    sample_dir = os.path.join(ROOT, 'tests', 'samples')
    if os.path.isdir(sample_dir):
        files = sorted(os.listdir(sample_dir))
        if files:
            st.write('可用样例：' + '、'.join(files))
