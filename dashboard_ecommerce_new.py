import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score

# Configuração da página
st.set_page_config(page_title="E-Commerce Executive Analytics", layout="wide")

# --- CARREGAMENTO E TRATAMENTO DE DADOS ---
@st.cache_data
def load_and_process_data():
    df = pd.read_excel('E-Commerce Orders.csv.xlsx')
    
    # Tratamento de Datas e Ordenação Temporal
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['DayNum'] = df['Date'].dt.dayofweek 
    
    # Estatísticas por Cliente
    customer_stats = df.groupby('CustomerID').agg(
        CLV=('TotalPrice', 'sum'),
        OrderCount=('OrderID', 'count'),
        AvgTicket=('TotalPrice', 'mean')
    ).reset_index()
    
    # Segmentação por Quartis
    customer_stats['Segment'] = pd.qcut(customer_stats['CLV'], 4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    customer_stats['Segment'] = pd.Categorical(customer_stats['Segment'], 
                                              categories=['Platinum', 'Gold', 'Silver', 'Bronze'], 
                                              ordered=True)
    
    df = df.merge(customer_stats[['CustomerID', 'CLV', 'OrderCount', 'Segment']], on='CustomerID', how='left')
    return df, customer_stats

try:
    df, customer_stats = load_and_process_data()
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# --- INTERFACE ---
st.title("🚀 Business Intelligence & Predictive Analytics")

# --- ABAS RENOMEADAS ---
tab_diag, tab_cust, tab_mkt_op, tab_prod_saz, tab_temp, tab_ml = st.tabs([
    "🔍 Diagnosis", "👥 Customers", "📢 Marketing and Operations", "📦 Products and Seasonality", "📅 Time Series", "🤖 Modeling"
])

with tab_diag:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing Data Analysis (%)")
        missing_data = df.isnull().mean() * 100
        st.plotly_chart(px.bar(x=missing_data.index, y=missing_data.values, labels={'x':'Columns','y':'% Missing'}), use_container_width=True)
    with col2:
        st.subheader("Correlation Matrix")
        num_cols = ['Quantity', 'UnitPrice', 'ItemsInCart', 'TotalPrice', 'CLV', 'OrderCount']
        corr = df[num_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu', range_color=[-1, 1])
        st.plotly_chart(fig_corr, use_container_width=True)

with tab_cust:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Spending Distribution (CLV)")
        st.plotly_chart(px.histogram(customer_stats, x="CLV", nbins=50), use_container_width=True)
    with col_c2:
        st.subheader("Purchase Frequency")
        st.plotly_chart(px.histogram(customer_stats, x="OrderCount"), use_container_width=True)
    st.subheader("CLV by Segment")
    st.plotly_chart(px.box(customer_stats, x='Segment', y='CLV', color='Segment',
                           category_orders={"Segment": ["Platinum", "Gold", "Silver", "Bronze"]}), use_container_width=True)

with tab_mkt_op:
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Coupon Participation (+ No Coupon)")
        coupon_df = df['CouponCode'].fillna('No Coupon').value_counts(normalize=True).reset_index()
        coupon_df.columns = ['Coupon', 'Proportion']
        st.plotly_chart(px.pie(coupon_df, names='Coupon', values='Proportion', hole=0.3), use_container_width=True)
    with col_m2:
        st.subheader("Cancellation Rate by Channel (%)")
        can_rate = df.groupby('ReferralSource')['OrderStatus'].apply(lambda x: (x == 'Cancelled').mean() * 100).reset_index()
        can_rate.columns = ['ReferralSource', 'CancellationRate']
        can_rate = can_rate.sort_values('CancellationRate', ascending=False)
        st.plotly_chart(px.bar(can_rate, x='ReferralSource', y='CancellationRate', color='CancellationRate'), use_container_width=True)
    
    st.divider()
    st.subheader("General Order Status")
    status_share = df['OrderStatus'].value_counts(normalize=True).reset_index()
    status_share.columns = ['Status', 'Proportion']
    st.plotly_chart(px.pie(status_share, names='Status', values='Proportion', title="Status Share (Operations)"), use_container_width=True)

with tab_prod_saz:
    st.subheader("Product Mix Analysis")
    prod_stats = df.groupby('Product').agg({'TotalPrice': 'sum', 'OrderID': 'count'}).reset_index()
    prod_stats['Average Revenue'] = prod_stats['TotalPrice'] / prod_stats['OrderID']
    p1, p2, p3 = st.columns(3)
    p1.plotly_chart(px.pie(prod_stats, names='Product', values='TotalPrice', title="% Revenue"), use_container_width=True)
    p2.plotly_chart(px.pie(prod_stats, names='Product', values='OrderID', title="% Orders"), use_container_width=True)
    p3.plotly_chart(px.bar(prod_stats.sort_values('Average Revenue', ascending=False), x='Product', y='Average Revenue', title="Average Revenue"), use_container_width=True)
    
    st.divider()
    
    col_st1, col_st2 = st.columns(2)
    with col_st1:
        st.subheader("Weekly Seasonality (Index)")
        df['WeekIdx'] = df['Date'].dt.isocalendar().week
        week_sales = df.groupby(['Year', 'WeekIdx', 'DayOfWeek', 'DayNum'])['TotalPrice'].sum().reset_index()
        avg_week = week_sales.groupby(['Year', 'WeekIdx'])['TotalPrice'].transform('mean')
        week_sales['Idx'] = week_sales['TotalPrice'] / avg_week
        season_week = week_sales.groupby(['DayOfWeek', 'DayNum'])['Idx'].mean().reset_index().sort_values('DayNum')
        st.plotly_chart(px.line(season_week, x='DayOfWeek', y='Idx', markers=True), use_container_width=True)

    with col_st2:
        st.subheader("Monthly Seasonality (Performance vs. Annual Average)")
        m_sales = df.groupby(['Year', 'Month'])['TotalPrice'].sum().reset_index()
        y_avg = m_sales.groupby('Year')['TotalPrice'].transform('mean')
        m_sales['Index_Month'] = m_sales['TotalPrice'] / y_avg
        season_month = m_sales.groupby('Month')['Index_Month'].mean().reset_index()
        fig_sm = px.line(season_month, x='Month', y='Index_Month', markers=True)
        fig_sm.add_hline(y=1.0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_sm, use_container_width=True)

with tab_temp:
    st.subheader("Historical Revenue Evolution")
    
    # Mensal
    st.write("#### Monthly Revenue (6-month MA)")
    m_ts = df.set_index('Date').resample('ME')['TotalPrice'].sum().reset_index()
    m_ts['MM6'] = m_ts['TotalPrice'].rolling(6).mean()
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(x=m_ts['Date'], y=m_ts['TotalPrice'], name='Monthly'))
    fig_m.add_trace(go.Scatter(x=m_ts['Date'], y=m_ts['MM6'], name='6-Month MA', line=dict(dash='dash')))
    st.plotly_chart(fig_m, use_container_width=True)

    # Quarter
    st.write("#### Quarterly Revenue (4-period MA)")
    q_ts = df.set_index('Date').resample('QE')['TotalPrice'].sum().reset_index()
    q_ts['MM4'] = q_ts['TotalPrice'].rolling(4).mean()
    fig_q = go.Figure()
    fig_q.add_trace(go.Scatter(x=q_ts['Date'], y=q_ts['TotalPrice'], name='Quarterly', mode='lines+markers'))
    fig_q.add_trace(go.Scatter(x=q_ts['Date'], y=q_ts['MM4'], name='4-Quarter MA', line=dict(dash='dot', color='orange')))
    st.plotly_chart(fig_q, use_container_width=True)

    # Anual
    st.write("#### Annual Evolution")
    y_ts = df.groupby('Year')['TotalPrice'].sum().reset_index()
    st.plotly_chart(px.bar(y_ts, x='Year', y='TotalPrice', text_auto='.2s', color='TotalPrice'), use_container_width=True)

with tab_ml:
    st.subheader("🤖 Driver Modeling with Temporal Validation")
    
    df_ml = df.copy().dropna(subset=['ReferralSource', 'PaymentMethod', 'CouponCode'])
    le = LabelEncoder()
    df_ml['Ref_Enc'] = le.fit_transform(df_ml['ReferralSource'])
    df_ml['Pay_Enc'] = le.fit_transform(df_ml['PaymentMethod'])
    df_ml['Coup_Enc'] = le.fit_transform(df_ml['CouponCode'])
    df_ml['IsCancelled'] = (df_ml['OrderStatus'] == 'Cancelled').astype(int)
    
    features = ['Quantity', 'UnitPrice', 'ItemsInCart', 'Ref_Enc', 'Pay_Enc', 'Coup_Enc']
    X = df_ml[features]
    
    # Split Temporal (80/20)
    split_idx = int(len(df_ml) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    
    # Modelo de Revenue
    y_rev = df_ml['TotalPrice']
    reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_rev.iloc[:split_idx])
    mae = mean_absolute_error(y_rev.iloc[split_idx:], reg.predict(X_test))
    
    # Modelo de Cancelamento
    y_can = df_ml['IsCancelled']
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_can.iloc[:split_idx])
    acc = accuracy_score(y_can.iloc[split_idx:], clf.predict(X_test))
    
    m1, m2 = st.columns(2)
    m1.metric("Mean Error (MAE) - Revenue", f"R$ {mae:.2f}")
    m2.metric("Accuracy - Cancellation", f"{acc:.2%}")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.write("### Importance: Revenue")
        imp_rev = pd.DataFrame({'Feature': features, 'Importancia': reg.feature_importances_}).sort_values('Importancia')
        st.plotly_chart(px.bar(imp_rev, x='Importancia', y='Feature', orientation='h', color_discrete_sequence=['#00CC96']), use_container_width=True)
    with col_g2:
        st.write("### Importance: Cancellation")
        imp_can = pd.DataFrame({'Feature': features, 'Importancia': clf.feature_importances_}).sort_values('Importancia')
        st.plotly_chart(px.bar(imp_can, x='Importancia', y='Feature', orientation='h', color_discrete_sequence=['#EF553B']), use_container_width=True)
