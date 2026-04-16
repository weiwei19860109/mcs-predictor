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
    st.metric("AUC", "0.670 (95% CI: 0.565-0.770)")
    st.metric("准确率", "63.89%")
    st.metric("敏感度", "87.27%")
    st.metric("特异度", "39.62%")
    st.metric("最佳阈值", "0.3802")

# 使用columns布局避免重复ID
col1, col2 = st.columns(2)

with col1:
    st.subheader("Alpha频段 (8-13Hz)")
    # 为每个slider添加唯一的key
    a_clust = st.slider(
        "聚类系数", 
        0.0, 1.0, 0.5, 0.01,
        key="alpha_clustering",
        help="网络局部连接密度，值越高表示局部连接越紧密"
    )
    a_deg = st.slider(
        "度中心性均值", 
        0.0, 1.0, 0.5, 0.01,
        key="alpha_degree",
        help="节点连接强度的平均值"
    )
    a_plv = st.slider(
        "PLV均值", 
        0.0, 1.0, 0.5, 0.01,
        key="alpha_plv",
        help="相位锁定值均值，反映功能连接强度"
    )

with col2:
    st.subheader("Beta频段 (13-30Hz)")
    b_deg = st.slider(
        "度中心性均值", 
        0.0, 1.0, 0.5, 0.01,
        key="beta_degree",
        help="节点连接强度的平均值"
    )
    b_plv = st.slider(
        "PLV均值", 
        0.0, 1.0, 0.5, 0.01,
        key="beta_plv",
        help="相位锁定值均值，反映功能连接强度"
    )

# 预测按钮
if st.button("🔍 预测MCS概率", type="primary", use_container_width=True):
    prob, logit = predict(a_clust, a_deg, a_plv, b_deg, b_plv)
    
    st.markdown("---")
    st.subheader("📊 预测结果")
    
    # 使用三列布局
    col3, col4, col5 = st.columns([2, 1, 1])
    
    with col3:
        st.metric("MCS概率", f"{prob:.2%}")
        # 使用进度条显示概率
        st.progress(prob)
    
    with col4:
        if prob >= THRESHOLD:
            st.success(f"✅ MCS阳性")
            st.caption(f"概率 ≥ {THRESHOLD:.1%}")
        else:
            st.error(f"❌ MCS阴性")
            st.caption(f"概率 < {THRESHOLD:.1%}")
    
    with col5:
        st.metric("Logit值", f"{logit:.3f}")
    
    # 可展开的详细计算
    with st.expander("📐 查看计算公式详情"):
        st.markdown("**模型公式:**")
        st.latex(r"Logit(P) = -4.447 + 45.092 \times \alpha_{clust}")
        st.latex(r"- 1.154 \times \alpha_{deg} - 20.777 \times \alpha_{plv}")
        st.latex(r"+ 0.279 \times \beta_{deg} + 5.027 \times \beta_{plv}")
        
        st.markdown("**当前计算过程:**")
        st.write(f"α_clust = {a_clust:.3f}, α_deg = {a_deg:.3f}, α_plv = {a_plv:.3f}")
        st.write(f"β_deg = {b_deg:.3f}, β_plv = {b_plv:.3f}")
        st.write(f"**Logit(P) = {logit:.4f}**")
        st.write(f"**P(MCS) = 1/(1+e^({-logit:.4f})) = {prob:.4f} ({prob:.2%})**")
    
    # 临床建议
    st.info("⚠️ 预测结果仅供参考，请结合临床专业判断")

# 添加使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 如何使用
    1. 在左侧输入5个脑网络特征参数（范围0-1）
    2. 点击"预测MCS概率"按钮
    3. 查看预测结果
    
    ### 特征说明
    - **聚类系数**: 衡量网络局部连接密度
    - **度中心性均值**: 节点连接强度的平均值
    - **PLV均值**: 相位锁定值均值，反映功能连接强度
    
    ### 模型解读
    - **阳性（MCS）**: 预测概率 ≥ 0.3802
    - **阴性（非MCS）**: 预测概率 < 0.3802
    
    ### 模型性能
    - AUC: 0.670 (95% CI: 0.565-0.770)
    - 准确率: 63.89%
    - 敏感度: 87.27% (能正确识别87.27%的MCS患者)
    - 特异度: 39.62%
    """)
streamlit
numpy
