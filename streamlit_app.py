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

# 主标题
st.title("🧠 MCS预测验证系统")
st.markdown("基于脑网络特征的微意识状态预测")

# 侧边栏
with st.sidebar:
    st.header("📊 模型性能")
    st.metric("AUC", "0.670 (95% CI: 0.565-0.770)")
    st.metric("准确率", "63.89%")
    st.metric("敏感度", "87.27%")
    st.metric("特异度", "39.62%")
    st.metric("最佳阈值", "0.3802")
    
    st.markdown("---")
    st.markdown("**模型公式:**")
    st.latex(r"Logit(P) = -4.447 + 45.092 \times \alpha_{clust}")
    st.latex(r"- 1.154 \times \alpha_{deg} - 20.777 \times \alpha_{plv}")
    st.latex(r"+ 0.279 \times \beta_{deg} + 5.027 \times \beta_{plv}")

# 使用columns布局
col1, col2 = st.columns(2)

with col1:
    st.subheader("Alpha频段 (8-13Hz)")
    
    # 使用数字输入框替代滑块
    a_clust = st.number_input(
        "聚类系数",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%.4f",
        key="alpha_clustering_input",
        help="网络局部连接密度，值越高表示局部连接越紧密"
    )
    
    a_deg = st.number_input(
        "度中心性均值",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%.4f",
        key="alpha_degree_input",
        help="节点连接强度的平均值"
    )
    
    a_plv = st.number_input(
        "PLV均值",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%.4f",
        key="alpha_plv_input",
        help="相位锁定值均值，反映功能连接强度"
    )

with col2:
    st.subheader("Beta频段 (13-30Hz)")
    
    b_deg = st.number_input(
        "度中心性均值",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%.4f",
        key="beta_degree_input",
        help="节点连接强度的平均值"
    )
    
    b_plv = st.number_input(
        "PLV均值",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%.4f",
        key="beta_plv_input",
        help="相位锁定值均值，反映功能连接强度"
    )

# 快速预设按钮
st.markdown("---")
st.markdown("**快速预设:**")
preset_cols = st.columns(3)

with preset_cols[0]:
    if st.button("📊 平均特征", use_container_width=True, key="preset_avg"):
        st.session_state.alpha_clustering_input = 0.5
        st.session_state.alpha_degree_input = 0.5
        st.session_state.alpha_plv_input = 0.5
        st.session_state.beta_degree_input = 0.5
        st.session_state.beta_plv_input = 0.5
        st.rerun()

with preset_cols[1]:
    if st.button("⬆️ 高风险预设", use_container_width=True, key="preset_high"):
        st.session_state.alpha_clustering_input = 0.8
        st.session_state.alpha_degree_input = 0.3
        st.session_state.alpha_plv_input = 0.2
        st.session_state.beta_degree_input = 0.7
        st.session_state.beta_plv_input = 0.8
        st.rerun()

with preset_cols[2]:
    if st.button("⬇️ 低风险预设", use_container_width=True, key="preset_low"):
        st.session_state.alpha_clustering_input = 0.2
        st.session_state.alpha_degree_input = 0.8
        st.session_state.alpha_plv_input = 0.8
        st.session_state.beta_degree_input = 0.3
        st.session_state.beta_plv_input = 0.2
        st.rerun()

# 预测按钮
st.markdown("---")
if st.button("🔍 预测MCS概率", type="primary", use_container_width=True, key="predict_button"):
    # 获取当前输入值
    current_a_clust = st.session_state.get('alpha_clustering_input', 0.5)
    current_a_deg = st.session_state.get('alpha_degree_input', 0.5)
    current_a_plv = st.session_state.get('alpha_plv_input', 0.5)
    current_b_deg = st.session_state.get('beta_degree_input', 0.5)
    current_b_plv = st.session_state.get('beta_plv_input', 0.5)
    
    prob, logit = predict(current_a_clust, current_a_deg, current_a_plv, 
                         current_b_deg, current_b_plv)
    
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
        st.markdown("**当前输入参数:**")
        st.write(f"• α_clust (Alpha聚类系数) = {current_a_clust:.4f}")
        st.write(f"• α_deg (Alpha度中心性均值) = {current_a_deg:.4f}")
        st.write(f"• α_plv (Alpha PLV均值) = {current_a_plv:.4f}")
        st.write(f"• β_deg (Beta度中心性均值) = {current_b_deg:.4f}")
        st.write(f"• β_plv (Beta PLV均值) = {current_b_plv:.4f}")
        
        st.markdown("**计算过程:**")
        st.write(f"Logit(P) = {INTERCEPT:.6f}")
        st.write(f"  + 45.092475 × {current_a_clust:.4f} = {45.092475 * current_a_clust:.6f}")
        st.write(f"  - 1.154255 × {current_a_deg:.4f} = {-1.154255 * current_a_deg:.6f}")
        st.write(f"  - 20.776595 × {current_a_plv:.4f} = {-20.776595 * current_a_plv:.6f}")
        st.write(f"  + 0.279283 × {current_b_deg:.4f} = {0.279283 * current_b_deg:.6f}")
        st.write(f"  + 5.027090 × {current_b_plv:.4f} = {5.027090 * current_b_plv:.6f}")
        st.write(f"**Logit(P) = {logit:.6f}**")
        st.write(f"**P(MCS) = 1/(1+e^(-{logit:.6f})) = {prob:.6f} ({prob:.2%})**")
    
    # 临床建议
    st.info("⚠️ 预测结果仅供参考，请结合临床专业判断")

# 添加使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 如何使用
    1. 在左侧输入5个脑网络特征参数（范围0-1）
    2. 可以使用快速预设按钮快速填充示例数据
    3. 点击"预测MCS概率"按钮
    4. 查看预测结果和详细计算过程
    
    ### 特征说明
    - **聚类系数**: 衡量网络局部连接密度，值越高表示局部连接越紧密
    - **度中心性均值**: 节点连接强度的平均值，反映网络整体连接水平
    - **PLV均值**: 相位锁定值均值，反映功能连接强度
    
    ### 模型解读
    - **阳性（MCS）**: 预测概率 ≥ 0.3802
    - **阴性（非MCS）**: 预测概率 < 0.3802
    
    ### 模型性能
    - **AUC**: 0.670 (95% CI: 0.565-0.770) - 模型区分能力中等
    - **准确率**: 63.89% - 整体预测准确率
    - **敏感度**: 87.27% - 能正确识别87.27%的MCS患者
    - **特异度**: 39.62% - 能正确识别39.62%的非MCS患者
    
    ### 注意事项
    - 所有特征参数应在0-1范围内
    - 预测结果仅供参考，不能替代临床诊断
    - 如有疑问，请咨询专业医生
    """)

# 页脚
st.markdown("---")
st.caption("© 2025 MCS预测验证系统 | 基于逻辑回归模型 | 版本 1.0")
streamlit
numpy
