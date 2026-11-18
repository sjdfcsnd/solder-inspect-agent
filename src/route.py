# -*- coding: utf-8 -*-
"""置信度路由与人工复核规则。

路由要同时解决两个问题：低置信度不能直接下结论；高影响类别（桥连，漏检等于放过
短路）即使置信度较高也应让人确认。因此规则由置信度阈值、类别风险与类别间差距三
部分组成，而不是单看 argmax。

复核队列的取舍还考虑误判代价的不对称：把正常件误判为缺陷只是多花一次人工确认，
而把缺陷放行会流出不良品。因此判为缺陷的样本进入复核的门槛更严（阈值更高），
判为正常的样本只有在该判断本身极不确定时才复核，避免复核资源被大量正常件占满。
"""

import numpy as np

HIGH_IMPACT = 1  # bridge
CLASS_CN = ['正常', '桥连', '虚焊少锡', '多锡', '偏移']
NORMAL = 0


def decide(probs, th=0.35, high_th=0.80, gap_min=0.10, normal_th=0.30):
    probs = np.asarray(probs, dtype=float)
    order = probs.argsort()[::-1]
    top, second = int(order[0]), int(order[1])
    conf = float(probs[top])
    gap = float(probs[top] - probs[second])

    if top == NORMAL:
        if conf < normal_th:
            return {'label': top, 'conf': conf, 'action': 'review',
                    'reason': '判为正常但置信度仅 %.2f，需确认是否漏检' % conf}
        return {'label': top, 'conf': conf, 'action': 'auto',
                'reason': '判为正常且置信度 %.2f' % conf}

    if conf < th:
        return {'label': top, 'conf': conf, 'action': 'review',
                'reason': '缺陷判定置信度低（%.2f < %.2f）' % (conf, th)}
    if top == HIGH_IMPACT and conf < high_th:
        return {'label': top, 'conf': conf, 'action': 'review',
                'reason': '高影响类别桥连需人工确认（%.2f < %.2f）' % (conf, high_th)}
    if gap < gap_min:
        return {'label': top, 'conf': conf, 'action': 'review',
                'reason': '类别竞争（与次优差 %.2f）' % gap}
    return {'label': top, 'conf': conf, 'action': 'auto', 'reason': '规则内自动出结论'}
