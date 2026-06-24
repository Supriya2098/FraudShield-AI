import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


sys.path.append("..")
from src.predict import FraudPredictor

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
background:#0A192F;
color:#CCD6F6;
}

section[data-testid="stSidebar"]{
    background:#112240;
}

section[data-testid="stSidebar"] h1{
    color:#00E5FF !important;
}
label[data-testid="stWidgetLabel"] p{
    color:white !important;
    font-weight:600 !important;
}
.stRadio label{
    color:white !important;
}

.stRadio label p{
    color:white !important;
    opacity:1 !important;
}




.metric-card{
background:linear-gradient(135deg,#112240,#1A365D);
padding:20px;
border-radius:20px;
border-left:4px solid #00E5FF;
height:140px;
transition:all 0.3s ease;
box-shadow:0 4px 20px rgba(0,229,255,0.15);
}

.metric-card:hover{
    transform:translateY(-8px);
    box-shadow:0 10px 30px rgba(0,229,255,0.35);
}

h1,h2,h3,h4{
color:#00E5FF !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
   

    

    DATA_PATH = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "feature_engineered_transactions.csv"
    )

    return pd.read_csv(DATA_PATH)

def metric_card(title,value):
    st.markdown(f"""
    <div class="metric-card">
    <h4>{title}</h4>
    <h2>{value}</h2>
    </div>
    """, unsafe_allow_html=True)

def style_plot(fig):
    fig.update_layout(
        paper_bgcolor="#112240",
        plot_bgcolor="#112240",
        font_color="white",
        height=450
    )
    return fig

df = load_data()


if os.path.exists("../assets/logo.jpeg"):
    st.sidebar.image(
        "../assets/logo.jpeg",
        width=180
    )
st.sidebar.title("🛡️ FraudShield AI")    



selected_country = st.sidebar.selectbox(
    "🌍 Select Country",
    ["All Countries"] + sorted(df["country"].unique())
)

if selected_country != "All Countries":
    filtered_df = df[df["country"] == selected_country]
else:
    filtered_df = df.copy()



menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard","Analytics","Live Prediction","Batch Prediction","Explainable AI","Reports","About"]
)

if menu == "Dashboard":
    st.markdown("""
    <div style="
    padding:30px;
    border-radius:25px;
    background:linear-gradient(
    135deg,
    #00E5FF,
    #2563EB
    );
    text-align:center;
    color:white;
    margin-bottom:25px;
    ">
    <h1 style="
    color:white;
    text-shadow:0px 2px 8px rgba(0,0,0,0.3);
    ">
    🛡️ FraudShield AI
    </h1>

    <h3 style="
    color:white;
    text-shadow:0px 2px 8px rgba(0,0,0,0.3);
    ">
    Enterprise Fraud Intelligence Platform
    </h3>
    </div>
    """, unsafe_allow_html=True)
    st.title("🛡️ FraudShield AI")
    st.caption("Global Fraud Intelligence & Detection Platform")

    total_txn = len(filtered_df)
    fraud_txn = int(filtered_df["is_fraud"].sum())
    fraud_rate = round((fraud_txn/total_txn)*100,2)
    risk_score = round(fraud_rate*2.5,2)

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Transactions", f"{total_txn:,}")
    with c2: metric_card("Fraud Cases", f"{fraud_txn:,}")
    with c3: metric_card("Fraud Rate", f"{fraud_rate}%")
    with c4: metric_card("Risk Score", risk_score)
    
    st.markdown("## 📌 Executive Summary")

    colA,colB,colC = st.columns(3)

    with colA:
        st.info(f"Total Transactions: {total_txn:,}")

    with colB:
        st.warning(f"Fraud Cases: {fraud_txn:,}")

    with colC:
        st.success(f"Fraud Rate: {fraud_rate}%")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fraud_rate,
        title={"text":"Fraud Risk %"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"#FF5252"},
            "steps":[
                {"range":[0,30],"color":"green"},
                {"range":[30,60],"color":"yellow"},
                {"range":[60,100],"color":"red"}
            ]
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    fraud_country = filtered_df.groupby("country")["is_fraud"].mean().reset_index()

    fig_map = px.choropleth(
        fraud_country,
        locations="country",
        locationmode="country names",
        color="is_fraud",
        title="Global Fraud Distribution"
    )
    st.plotly_chart(style_plot(fig_map), use_container_width=True)
    st.subheader("🏆 Top High-Risk Countries")
    
    leaderboard = (
        fraud_country
        .sort_values("is_fraud", ascending=False)
        .head(5)
    )

    for i, row in enumerate(
         leaderboard.itertuples(),
        start=1
    ):
        st.warning(
            f"#{i}  {row.country}  | Fraud Score: {row.is_fraud:.2%}"
        )


    col1,col2 = st.columns(2)

    fig1 = px.bar(
        fraud_country,
        x="country",
        y="is_fraud",
        color="is_fraud",
        color_continuous_scale="Turbo",
        title="Fraud Rate by Country"
    )

    fig2 = px.bar(
        filtered_df.groupby("device_type")["is_fraud"]
        .mean()
        .reset_index(),
        x="device_type",
        y="is_fraud",
        color="is_fraud",
        color_continuous_scale="Turbo",
        title="Fraud Rate by Device"
    )

    with col1:
        st.plotly_chart(style_plot(fig1), use_container_width=True)
    with col2:
        st.plotly_chart(style_plot(fig2), use_container_width=True)

    st.subheader("🤖 AI Insights")
    highest_country = fraud_country.sort_values("is_fraud", ascending=False).iloc[0]
    st.success(f"Highest fraud risk country: {highest_country['country']}")
    st.info(f"Overall fraud rate detected: {fraud_rate:.2f}%")
    st.subheader("🚨 Live Fraud Alerts")

    alerts = (
        filtered_df[
            filtered_df["is_fraud"] == 1
        ]
        .sort_values(
            "transaction_amount",
            ascending=False
        )
        .head(10)
    )

    for _, row in alerts.iterrows():
        st.error(
            f"⚠️ HIGH RISK | {row['country']} | Amount: ${row['transaction_amount']:.2f}"
        )

elif menu == "Analytics":
    st.title("📈 Analytics")

    selected_country = st.selectbox(
        "Select Country",
        sorted(df["country"].unique())
    )

    country_df = df[df["country"] == selected_country]

    tab1,tab2,tab3 = st.tabs(["Distribution","Timeline","Top Frauds"])

    with tab1:
        fig = px.histogram(
            country_df,
            x="transaction_amount",
            color="is_fraud",
            color_discrete_sequence=[
                "#00E5FF",
                "#EF4444"
            ]
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with tab2:
        fig = px.line(
            country_df.groupby("transaction_month")["is_fraud"]
            .sum()
            .reset_index(),
            x="transaction_month",
            y="is_fraud",
            markers=True
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with tab3:
        suspicious = country_df[
            (country_df["transaction_amount"] > 500) &
            (country_df["is_fraud"] == 1)
        ]
        st.dataframe(suspicious.head(20), use_container_width=True)

elif menu == "Live Prediction":
    st.title("🔍 Live Prediction")
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    hour = st.slider("Transaction Hour",0,23,12)

    if st.button("Predict Risk"):
        if amount > 300:
            st.error("⚠️ High Fraud Risk")
        else:
            st.success("✅ Low Fraud Risk")

elif menu == "Batch Prediction":
    st.title("📂 Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        upload_df = pd.read_csv(uploaded_file)
        MODEL_PATH = os.path.join(
            BASE_DIR,
            "models",
            "random_forest.pkl"
        )

        predictor = FraudPredictor(MODEL_PATH)
        
        results = predictor.batch_predict(upload_df)

        st.dataframe(results, use_container_width=True)

        st.download_button(
            "Download Predictions",
            results.to_csv(index=False).encode("utf-8"),
            "fraud_predictions.csv"
        )

elif menu == "Explainable AI":
    st.title("🧠 Explainable AI")

    tab1,tab2,tab3 = st.tabs(
        ["Feature Importance","SHAP Summary","Model Insights"]
    )

    with tab1:
        if os.path.exists("../reports/feature_importance.png"):
            st.image("../reports/feature_importance.png", use_container_width=True)

    with tab2:
        if os.path.exists("../reports/shap_summary_fixed.png"):
            st.image("../reports/shap_summary_fixed.png", use_container_width=True)

    with tab3:
        if os.path.exists("../reports/shap_bar_fixed.png"):
            st.image("../reports/shap_bar_fixed.png", use_container_width=True)

elif menu == "Reports":
    st.title("📄 Reports")

    for file in [
        "../reports/fraud_summary.csv",
        "../reports/fraud_predictions.csv",
        "../reports/top_fraud_cases.csv"
    ]:
        if os.path.exists(file):
            st.markdown(f"### 📄 {os.path.basename(file)}")
            with open(file, "rb") as f:
                st.download_button(
                    f"Download {os.path.basename(file)}",
                    f.read(),
                    os.path.basename(file)
                )

elif menu == "About":
    st.title("🛡️ FraudShield AI")

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Accuracy","97.86%")
    with c2: st.metric("Precision","70.19%")
    with c3: st.metric("Recall","80.65%")

    st.metric("ROC-AUC","98.95%")
    st.metric("F1 Score","75.08%")

    st.markdown("""
### Enterprise Fraud Detection Platform

Technologies:
- Python
- Pandas
- Scikit-Learn
- SHAP
- Plotly
- Streamlit

Developer: Supriya Kusuma
""")
