import streamlit as st
import pandas as pd
import numpy as np
from app_tier_classifier import RedesignedAppTierClassifier
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Apponomics - Redesigned Tier Classification",
    page_icon="📱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .framework-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
        color: #000000 !important;
    }
    .framework-box h3, .framework-box p {
        color: #000000 !important;
    }
    .neutral-apps {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
        color: #000000 !important;
    }
    .neutral-apps h4, .neutral-apps p, .neutral-apps li {
        color: #000000 !important;
    }
    .discriminator-apps {
        background-color: #fff3e0;
        border-left-color: #ff9800;
        color: #000000 !important;
    }
    .discriminator-apps h4, .discriminator-apps p, .discriminator-apps li {
        color: #000000 !important;
    }
    .tier-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
        color: #000000 !important;
    }
    .tier-card h4 {
        color: #000000 !important;
    }
    .premium-tier {
        border-left-color: #ff6b6b;
        background-color: #ffe0e0;
        color: #000000 !important;
    }
    .premium-tier h4 {
        color: #000000 !important;
    }
    .standard-tier {
        border-left-color: #4ecdc4;
        background-color: #e0f7f5;
        color: #000000 !important;
    }
    .standard-tier h4 {
        color: #000000 !important;
    }
    .basic-tier {
        border-left-color: #45b7d1;
        background-color: #e0f4ff;
        color: #000000 !important;
    }
    .basic-tier h4 {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📱 Apponomics</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">Redesigned App-Based Tier Classification</h2>', unsafe_allow_html=True)
    
    # Framework explanation
    st.markdown("""
    <div class="framework-box">
        <h3>🎯 New Framework: Neutral Apps vs Discriminator Apps</h3>
        <p><strong>Neutral Apps</strong> (Zomato, Paytm, Blinkit) are used across all tiers - behavior matters more than presence.</p>
        <p><strong>Discriminator Apps</strong> (CRED, Zerodha, Meesho, ShareChat) provide strong tier signals.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for input
    st.sidebar.header("📱 Enter Installed Apps")
    
    # App input
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Manual Entry", "Upload CSV", "Sample Profiles"]
    )
    
    installed_apps = []
    behavioral_data = {}
    
    if input_method == "Manual Entry":
        apps_text = st.sidebar.text_area(
            "Enter app names (one per line):",
            placeholder="zomato\nurban_company\nblinkit\ninstagram\nreddit\nunacademy\npaytm",
            height=200
        )
        if apps_text:
            installed_apps = [app.strip() for app in apps_text.split('\n') if app.strip()]
        
        # Behavioral data input
        st.sidebar.markdown("### 📊 Behavioral Data (Optional)")
        st.sidebar.markdown("Enter usage patterns for neutral apps:")
        
        if 'zomato' in installed_apps:
            zomato_orders = st.sidebar.number_input("Zomato orders per month", min_value=0, value=5)
            zomato_aov = st.sidebar.number_input("Zomato average order value (₹)", min_value=0, value=200)
            behavioral_data.update({
                'zomato_orders_per_month': zomato_orders,
                'zomato_avg_order_value': zomato_aov
            })
        
        if 'blinkit' in installed_apps:
            blinkit_orders = st.sidebar.number_input("Blinkit orders per month", min_value=0, value=4)
            blinkit_aov = st.sidebar.number_input("Blinkit average order value (₹)", min_value=0, value=180)
            behavioral_data.update({
                'blinkit_orders_per_month': blinkit_orders,
                'blinkit_avg_order_value': blinkit_aov
            })
        
        if 'paytm' in installed_apps:
            paytm_txn = st.sidebar.number_input("Paytm transactions per month", min_value=0, value=10)
            paytm_avg = st.sidebar.number_input("Paytm average transaction (₹)", min_value=0, value=300)
            behavioral_data.update({
                'paytm_txn_count': paytm_txn,
                'paytm_txn_avg_value': paytm_avg
            })
    
    elif input_method == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV with app names", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'apps' in df.columns:
                installed_apps = df['apps'].tolist()
            else:
                st.sidebar.error("CSV must contain an 'apps' column")
    
    else:  # Sample Profiles
        sample_options = {
            "Premium Urban Professional": {
                'apps': ['netflix', 'zerodha', 'airbnb', 'urban_company', 'linkedin_premium', 'cred'],
                'behavior': {'zomato_orders_per_month': 12, 'zomato_avg_order_value': 450, 'paytm_txn_avg_value': 800}
            },
            "Standard Middle-Class": {
                'apps': ['paytm', 'flipkart', 'youtube', 'instagram', 'unacademy', 'meesho'],
                'behavior': {'zomato_orders_per_month': 6, 'zomato_avg_order_value': 250, 'paytm_txn_avg_value': 400}
            },
            "Budget-Conscious User": {
                'apps': ['meesho', 'sharechat', 'ludo_king', 'kreditbee', 'cashbean', 'moj'],
                'behavior': {'zomato_orders_per_month': 3, 'zomato_avg_order_value': 150, 'paytm_txn_avg_value': 200}
            }
        }
        
        selected_sample = st.sidebar.selectbox("Choose sample profile:", list(sample_options.keys()))
        sample_data = sample_options[selected_sample]
        installed_apps = sample_data['apps']
        behavioral_data = sample_data['behavior']
        st.sidebar.write("Sample apps:", ", ".join(installed_apps))
    
    # Main content
    if installed_apps:
        st.markdown("---")
        
        # Initialize redesigned classifier
        classifier = RedesignedAppTierClassifier()
        
        # Analyze apps
        with st.spinner("Analyzing apps with redesigned framework..."):
            result = classifier.analyze_apps_with_behavior(installed_apps, behavioral_data)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 💰 Spending Tier")
            spending_tier = result['spending_tier']
            if "Premium" in spending_tier:
                st.markdown(f'<div class="tier-card premium-tier"><h4>{spending_tier}</h4></div>', unsafe_allow_html=True)
            elif "Standard" in spending_tier:
                st.markdown(f'<div class="tier-card standard-tier"><h4>{spending_tier}</h4></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="tier-card basic-tier"><h4>{spending_tier}</h4></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🏙️ Geographic Tier")
            geographic_tier = result['geographic_tier']
            if "Tier 1" in geographic_tier:
                st.markdown(f'<div class="tier-card premium-tier"><h4>{geographic_tier}</h4></div>', unsafe_allow_html=True)
            elif "Tier 2" in geographic_tier:
                st.markdown(f'<div class="tier-card standard-tier"><h4>{geographic_tier}</h4></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="tier-card basic-tier"><h4>{geographic_tier}</h4></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🎯 Lifestyle Category")
            lifestyle_category = result['lifestyle_category']
            if "Professional" in lifestyle_category:
                st.markdown(f'<div class="tier-card premium-tier"><h4>{lifestyle_category}</h4></div>', unsafe_allow_html=True)
            elif "Entertainment" in lifestyle_category:
                st.markdown(f'<div class="tier-card standard-tier"><h4>{lifestyle_category}</h4></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="tier-card basic-tier"><h4>{lifestyle_category}</h4></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Confidence and scores
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Analysis Confidence")
            confidence = result['confidence']
            st.progress(confidence / 100)
            st.write(f"Confidence: {confidence:.1f}%")
        
        with col2:
            st.metric("Spending Score", f"{result['spending_score']:.2f}")
        
        with col3:
            st.metric("Total Apps", len(installed_apps))
        
        # Detailed analysis
        st.markdown("---")
        st.markdown("### 🔍 Detailed Analysis")
        
        # Discriminator apps analysis
        st.markdown("#### 🎯 Discriminator Apps (Strong Tier Signals)")
        discriminator_analysis = result['discriminator_analysis']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Tier A Premium Apps:**")
            tier_a_apps = discriminator_analysis['matched_apps']['tier_a_premium']
            if tier_a_apps:
                for app in tier_a_apps:
                    st.write(f"✅ {app}")
            else:
                st.write("None found")
        
        with col2:
            st.markdown("**Tier B Mainstream Apps:**")
            tier_b_apps = discriminator_analysis['matched_apps']['tier_b_mainstream']
            if tier_b_apps:
                for app in tier_b_apps:
                    st.write(f"✅ {app}")
            else:
                st.write("None found")
        
        with col3:
            st.markdown("**Tier C Budget Apps:**")
            tier_c_apps = discriminator_analysis['matched_apps']['tier_c_budget']
            if tier_c_apps:
                for app in tier_c_apps:
                    st.write(f"✅ {app}")
            else:
                st.write("None found")
        
        # Neutral apps analysis
        st.markdown("#### ⚖️ Neutral Apps (Behavior Matters)")
        neutral_analysis = result['neutral_analysis']
        
        if neutral_analysis['usage_patterns']:
            usage_data = []
            for category, usage in neutral_analysis['usage_patterns'].items():
                usage_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Apps': ', '.join(usage['apps']),
                    'Frequency': f"{usage['frequency']:.0f}/month",
                    'AOV': f"₹{usage['aov']:.0f}",
                    'Monthly Spend': f"₹{usage['monthly_spend']:.0f}",
                    'Behavioral Score': f"{usage['behavioral_score']:.2f}"
                })
            
            usage_df = pd.DataFrame(usage_data)
            st.dataframe(usage_df, use_container_width=True)
        else:
            st.info("No neutral apps with behavioral data found.")
        
        # Visualizations
        st.markdown("---")
        st.markdown("### 📈 Analysis Visualizations")
        
        # Scores comparison
        scores_data = {
            "Metric": ["Spending Score", "Geographic Score", "Lifestyle Score"],
            "Score": [result['spending_score'], result['geographic_score'], result['lifestyle_score']]
        }
        
        fig_scores = px.bar(scores_data, x="Metric", y="Score", 
                           title="Tier Analysis Scores",
                           color="Score",
                           color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_scores, use_container_width=True)
        
        # Discriminator vs Neutral contribution
        contrib_data = {
            "Source": ["Discriminator Apps", "Neutral Apps"],
            "Spending Contribution": [
                discriminator_analysis['spending'],
                neutral_analysis['spending']
            ],
            "Geographic Contribution": [
                discriminator_analysis['geographic'],
                neutral_analysis['geographic']
            ],
            "Lifestyle Contribution": [
                discriminator_analysis['lifestyle'],
                neutral_analysis['lifestyle']
            ]
        }
        
        fig_contrib = px.bar(contrib_data, x="Source", 
                            y=["Spending Contribution", "Geographic Contribution", "Lifestyle Contribution"],
                            title="Score Contribution by App Type",
                            barmode='group')
        st.plotly_chart(fig_contrib, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Personalized Recommendations")
        
        recommendations = result['recommendations']
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
        
        # Export results
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        results_df = pd.DataFrame({
            "Metric": ["Spending Tier", "Geographic Tier", "Lifestyle Category", "Confidence", "Spending Score", "Geographic Score", "Lifestyle Score"],
            "Value": [result['spending_tier'], result['geographic_tier'], 
                     result['lifestyle_category'], f"{result['confidence']:.1f}%",
                     result['spending_score'], result['geographic_score'], result['lifestyle_score']]
        })
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Analysis Results",
            data=csv,
            file_name="redesigned_tier_analysis.csv",
            mime="text/csv"
        )
    
    else:
        st.info("👆 Please enter some apps in the sidebar to get started!")
        
        # Show framework explanation
        st.markdown("---")
        st.markdown("### 🎯 How the Redesigned Framework Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="framework-box neutral-apps">
                <h4>⚖️ Neutral Apps (Cross-tier)</h4>
                <p><strong>Examples:</strong> Zomato, Swiggy, Blinkit, Paytm, Uber, Amazon</p>
                <p><strong>Key Insight:</strong> Everyone uses these apps, but behavior differs:</p>
                <ul>
                    <li>Tier A: ₹450+ AOV, 12-15 orders/month</li>
                    <li>Tier B: ₹200-250 AOV, 3-5 orders/month</li>
                    <li>Tier C: ₹150 AOV, 1-2 orders/month</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="framework-box discriminator-apps">
                <h4>🎯 Discriminator Apps (Strong Signals)</h4>
                <p><strong>Tier A:</strong> CRED, Zerodha, Airbnb, Urban Company</p>
                <p><strong>Tier B:</strong> Meesho, Ajio, Unacademy, SonyLIV</p>
                <p><strong>Tier C:</strong> ShareChat, Ludo King, KreditBee, Moj</p>
                <p><strong>Key Insight:</strong> These apps strongly correlate with specific tiers</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show sample profiles
        st.markdown("### 📱 Sample Profiles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Premium Urban Professional:**")
            premium_apps = ["netflix", "zerodha", "airbnb", "urban_company", "linkedin_premium", "cred"]
            for app in premium_apps:
                st.write(f"• {app}")
            st.markdown("*High AOV, frequent usage*")
        
        with col2:
            st.markdown("**Standard Middle-Class:**")
            standard_apps = ["paytm", "flipkart", "youtube", "instagram", "unacademy", "meesho"]
            for app in standard_apps:
                st.write(f"• {app}")
            st.markdown("*Moderate AOV, regular usage*")
        
        with col3:
            st.markdown("**Budget-Conscious User:**")
            basic_apps = ["meesho", "sharechat", "ludo_king", "kreditbee", "cashbean", "moj"]
            for app in basic_apps:
                st.write(f"• {app}")
            st.markdown("*Low AOV, occasional usage*")

if __name__ == "__main__":
    main()
