# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 07:51:10 2026

@author: 没事不要上网
"""

import streamlit as st
import numpy as np

st.set_page_config(page_title="MCS预测系统", page_icon="🧠")

# 模型参数
INTERCEPT = -4.447085
THRESHOLD = 0.3802

def predict(a_clust, a_deg, a_plv, b_deg, b_plv):
    logit = INTERCEPT
    logit += 45.092475 * a_clust
    logit += -1.154255 * a_deg
    logit += -20.776595 * a_plv
    logit += 0.279283 * b_deg
    logit += 5.027090 * b_plv
    prob = 1 / (1 + np.exp(-logit))
    return prob, logit

st.title("🧠 MCS预测验证系统")
st.markdown("基于脑网络特征的微意识状态预测")

with st.sidebar:
    st.header("📊 模型性能")
    st.metric("AUC", "0.670")
    st.metric("准确率", "63.89%")
    st.metric("阈值", "0.3802")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Alpha频段")
    a_clust = st.slider("聚类系数", 0.0, 1.0, 0.5, 0.01)
    a_deg = st.slider("度中心性均值", 0.0, 1.0, 0.5, 0.01)
    a_plv = st.slider("PLV均值", 0.0, 1.0, 0.5, 0.01)

with col2:
    st.subheader("Beta频段")
    b_deg = st.slider("度中心性均值", 0.0, 1.0, 0.5, 0.01)
    b_plv = st.slider("PLV均值", 0.0, 1.0, 0.5, 0.01)

if st.button("🔍 预测", type="primary", use_container_width=True):
    prob, logit = predict(a_clust, a_deg, a_plv, b_deg, b_plv)
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.metric("MCS概率", f"{prob:.2%}")
        st.progress(prob)
    
    with col4:
        if prob >= THRESHOLD:
            st.success(f"✅ MCS阳性 (概率 ≥ {THRESHOLD:.1%})")
        else:
            st.error(f"❌ MCS阴性 (概率 < {THRESHOLD:.1%})")
    
    st.caption("⚠️ 预测结果仅供参考，请结合临床判断")
streamlit
numpy
