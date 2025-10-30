import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_processor import DataProcessor
from model_trainer import ModelTrainer
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Stop the Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': None,
        'About': '# Stop the Churn\nA customer churn prediction application.'
    }
)

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = ModelTrainer()
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_metadata' not in st.session_state:
    st.session_state.model_metadata = None

# Sidebar with enhanced styling and navigation
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: var(--primary-color); margin-bottom: 1rem;'>Stop the Churn</h1>
            <p style='color: var(--text-color);'>Upload your customer data to predict churn</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Privacy disclaimer
    st.markdown("---")
    with st.expander("🔒 Data Privacy & Security", expanded=False):
        st.warning("""
        **Your data privacy matters to us:**
        
        ✅ **Not Stored:** Uploaded data is processed in memory only
        ✅ **Not Shared:** Data never leaves your session
        ✅ **Secure:** No data persistence between sessions
        ✅ **Private:** Only you can see your results
        
        All analysis happens in real-time and data is cleared when you close the app.
        """)
    
    # File uploader with custom styling and help text
    uploaded_file = st.file_uploader(
        "Upload CSV file", 
        type=['csv', 'CSV'], 
        key="main_uploader",
        help="Upload a CSV file with customer data. The file should include a 'churn' column (0/1 or Yes/No). Maximum file size: 100 MB.",
        label_visibility="visible"
    )
    
    # Sidebar navigation
    st.markdown("---")
    st.markdown("### 📍 Navigation")
    page = st.radio(
        "Select a page",
        ["Overview", "Risk Analysis", "Feature Importance", "Real-time Prediction"],
        key="sidebar_nav",
        label_visibility="visible"
    )
    
    # Model metadata section
    if st.session_state.model_metadata is not None:
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        metadata = st.session_state.model_metadata
        
        # Responsive metadata display
        st.info(f"""
        **Training Date:** {metadata['training_date']}  
        **Dataset Version:** {metadata['dataset_version']}  
        **Total Records:** {metadata['total_records']:,}  
        **Model Type:** {metadata['model_type']}
        """)

if uploaded_file is not None:
    try:
        # Check file size (limit to 100MB) - if file already uploaded, check size attribute
        max_file_size = 100 * 1024 * 1024  # 100MB
        file_size = 0
        try:
            # Try to get file size from the uploaded file object
            if hasattr(uploaded_file, 'size'):
                file_size = uploaded_file.size
            elif hasattr(uploaded_file, 'getvalue'):
                file_size = len(uploaded_file.getvalue())
            
            if file_size > 0 and file_size > max_file_size:
                st.error(f"❌ File is too large ({file_size / (1024*1024):.2f} MB). Maximum allowed size is 100 MB.")
                st.warning("Please upload a smaller CSV file.")
                st.stop()
        except Exception as size_error:
            # If size check fails, proceed anyway (Streamlit will handle it)
            pass
        
        with st.spinner('Loading and processing data...'):
            df = pd.read_csv(uploaded_file)
            
            # Check if DataFrame is empty
            if df.empty:
                st.error("❌ The uploaded CSV file is empty. Please upload a file with data.")
                st.warning("Ensure your CSV file contains rows of data.")
                st.stop()
            
            # Check if DataFrame has columns
            if df.shape[1] == 0:
                st.error("❌ The uploaded CSV file has no columns. Please check the file format.")
                st.warning("Ensure your CSV file has column headers.")
                st.stop()
            
            # Success message
            st.success(f"✅ File uploaded successfully! Loaded {len(df)} rows and {len(df.columns)} columns.")
            
            # Display file info
            file_name = uploaded_file.name
            # Calculate file size for display
            if file_size > 0:
                file_size_mb = file_size / (1024 * 1024)
            else:
                file_size_mb = 0.0
            
            with st.expander("📄 File Information", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Filename", file_name)
                with col2:
                    st.metric("File Size", f"{file_size_mb:.2f} MB" if file_size > 0 else "N/A")
                with col3:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
                
                # Show column names
                st.markdown("**Columns in dataset:**")
                col_names = ", ".join(df.columns.tolist())
                st.code(col_names if len(col_names) < 500 else col_names[:500] + "...")
            
            st.session_state.data = df
            
            # Automatically convert 'churn' column to numeric if needed
            if 'churn' in df.columns:
                if df['churn'].dtype == object:
                    unique_vals = set(df['churn'].str.lower().unique())
                    if unique_vals == {'yes', 'no'} or unique_vals == {'no', 'yes'}:
                        df['churn'] = df['churn'].str.lower().map({'yes': 1, 'no': 0})
            else:
                # Warning if churn column is missing
                st.warning("⚠️ The uploaded file doesn't contain a 'churn' column. You can still preview the data, but model training requires a churn column.")
                st.info("💡 Add a 'churn' column with values 0 (no churn) and 1 (churned), or 'No' and 'Yes'.")
            
            # Data Preview with enhanced styling and pagination
            with st.expander("📊 Preview Uploaded Data", expanded=True):
                st.markdown(f"**Total Rows:** {len(df)} | **Total Columns:** {len(df.columns)}")
                
                # Show column info
                if len(df.columns) > 0:
                    col_types = df.dtypes.astype(str).to_dict()
                    col_info_text = " | ".join([f"**{col}**: {col_types[col]}" for col in df.columns[:5]])
                    if len(df.columns) > 5:
                        col_info_text += " ..."
                    st.caption(col_info_text)
                
                # Pagination controls
                rows_per_page = 10
                total_pages = (len(df) + rows_per_page - 1) // rows_per_page
                
                if total_pages > 1:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        page_num = st.number_input(
                            "Page",
                            min_value=1,
                            max_value=total_pages,
                            value=1,
                            step=1,
                            key="data_preview_page"
                        )
                    st.caption(f"Showing rows {(page_num-1)*rows_per_page + 1} to {min(page_num*rows_per_page, len(df))} of {len(df)}")
                else:
                    page_num = 1
                
                # Calculate slice indices
                start_idx = (page_num - 1) * rows_per_page
                end_idx = start_idx + rows_per_page
                
                # Display data
                if len(df.columns) <= 15:  # If reasonable number of columns, show full width table
                    st.dataframe(
                        df.iloc[start_idx:end_idx],
                        use_container_width=True,
                        height=400,
                        hide_index=False
                    )
                else:  # If too many columns, use scrolling
                    st.dataframe(
                        df.iloc[start_idx:end_idx],
                        use_container_width=True,
                        height=400,
                        hide_index=False
                    )
                    st.info(f"⚠️ Showing {len(df.columns)} columns. Scroll horizontally to view all.")
                
                # Quick stats (avoid nested expanders inside expander)
                st.markdown("#### 📈 Quick Statistics")
                st.dataframe(
                    df.describe(),
                    use_container_width=True
                )
            
            if 'churn' in df.columns:
                # CRITICAL: Split data BEFORE any preprocessing to prevent data leakage
                y = df['churn']
                X_raw = df.drop('churn', axis=1)
                
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X_raw, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Apply feature engineering only (no scaling/encoding yet)
                # Note: We'll apply this to train and test separately
                for df_temp in [X_train_raw, X_test_raw]:
                    if 'CustomerID' in df_temp.columns:
                        df_temp.drop('CustomerID', axis=1, inplace=True)
                
                # Fit preprocessor on TRAIN data only
                X_train_raw_with_churn = X_train_raw.copy()
                X_train_raw_with_churn['churn'] = y_train
                st.session_state.data_processor.fit(X_train_raw_with_churn)
                
                # Transform train and test data
                X_train_processed = st.session_state.data_processor.transform(X_train_raw_with_churn)
                X_train_processed = X_train_processed.drop('churn', axis=1)
                
                X_test_raw_with_churn = X_test_raw.copy()
                X_test_raw_with_churn['churn'] = y_test
                X_test_processed = st.session_state.data_processor.transform(X_test_raw_with_churn)
                X_test_processed = X_test_processed.drop('churn', axis=1)
                
                # Enhanced Stats Summary Cards
                total_customers = len(df)
                churned = int(df['churn'].sum())
                churn_rate = churned / total_customers if total_customers > 0 else 0
                
                # For display purposes, we'll use the full dataset
                # But for training, we use only train data
                X = pd.concat([X_train_processed, X_test_processed])
                y_full = pd.concat([y_train, y_test])
                
                # Mobile-responsive metrics display
                col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
                
                with col1:
                    st.markdown("""
                        <div class='metric-card animate-fade-in' role='region' aria-label='Total customers metric'>
                            <h3>Total Customers</h3>
                            <h2 style='color: var(--primary-color);' aria-label='{} customers'>{}</h2>
                        </div>
                    """.format(total_customers, total_customers), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class='metric-card animate-fade-in' role='region' aria-label='Churned customers metric'>
                            <h3>Churned</h3>
                            <h2 style='color: #ef5350;' aria-label='{} churned customers'>{}</h2>
                        </div>
                    """.format(churned, churned), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                        <div class='metric-card animate-fade-in' role='region' aria-label='Churn rate metric'>
                            <h3>Churn Rate</h3>
                            <h2 style='color: var(--secondary-color);' aria-label='{} percent'>{:.1%}</h2>
                        </div>
                    """.format(churn_rate * 100, churn_rate), unsafe_allow_html=True)
                
                with st.spinner('Training model and generating predictions...'):
                    # Train on TRAIN data only
                    results = st.session_state.model.train(X_train_processed, y_train)
                    
                    # Generate predictions for visualization (full dataset) but evaluate on TEST only
                    st.session_state.predictions = st.session_state.model.predict(X)
                    
                    # Get test set predictions for evaluation
                    test_predictions = st.session_state.model.predict(X_test_processed)
                    
                    # Calculate metrics on test set
                    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
                    test_auc = roc_auc_score(y_test, test_predictions)
                    test_f1 = f1_score(y_test, (test_predictions > 0.5).astype(int))
                    test_precision = precision_score(y_test, (test_predictions > 0.5).astype(int))
                    test_recall = recall_score(y_test, (test_predictions > 0.5).astype(int))
                    
                    # Store model metadata after metrics calculation
                    st.session_state.model_metadata = {
                        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'dataset_version': uploaded_file.name if hasattr(uploaded_file, 'name') else 'uploaded_data',
                        'total_records': len(df),
                        'model_type': 'Ensemble (RF + LGB + XGB)',
                        'test_auc': f'{test_auc:.4f}'
                    }
                    
                    # Model Evaluation - DISPLAY TEST SET METRICS
                    st.markdown("""
                        <div style='text-align: center; margin: 2rem 0;'>
                            <h2>Model Performance Metrics (Test Set Only)</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics from test set
                    st.info(f"**Test Set ROC AUC**: {test_auc:.4f} | **F1 Score**: {test_f1:.4f} | **Precision**: {test_precision:.4f} | **Recall**: {test_recall:.4f}")
                    
                    eval_col1, eval_col2 = st.columns(2)
                    with eval_col1:
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>Confusion Matrix (Test Set)</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        y_pred_test = (test_predictions > 0.5).astype(int)
                        cm = confusion_matrix(y_test, y_pred_test)
                        fig_cm = ff.create_annotated_heatmap(
                            cm,
                            x=["Predicted 0", "Predicted 1"],
                            y=["Actual 0", "Actual 1"],
                            colorscale='Blues',
                            showscale=True
                        )
                        fig_cm.update_layout(
                            template='plotly_white',
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with eval_col2:
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>ROC Curve (Test Set)</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        fpr, tpr, _ = roc_curve(y_test, test_predictions)
                        roc_auc_curve = auc(fpr, tpr)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            name=f'ROC curve (AUC = {roc_auc_curve:.2f})',
                            line=dict(color='#1E88E5', width=2)
                        ))
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            name='Random',
                            line=dict(color='#666', width=2, dash='dash')
                        ))
                        fig_roc.update_layout(
                            title='Receiver Operating Characteristic',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            showlegend=True,
                            template='plotly_white',
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                    
                    # Main content with tabs
                    st.markdown("""
                        <div style='text-align: center; margin: 2rem 0;'>
                            <h1>Churn Prediction Dashboard</h1>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📈 Overview",
                        "⚠️ Risk Analysis",
                        "🔍 Feature Importance",
                        "🎯 Real-time Prediction"
                    ])
                    
                    # Only show the selected section
                    if page == "Overview":
                        # Overview content (charts, metrics)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("""
                                <div class='stCard'>
                                    <h3 style='text-align: center;'>Churn Probability Distribution</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            fig = px.histogram(
                                x=st.session_state.predictions,
                                nbins=50,
                                title="Distribution of Churn Probabilities",
                                labels={'x': 'Churn Probability', 'y': 'Count'},
                                template='plotly_white'
                            )
                            fig.update_layout(
                                paper_bgcolor='white',
                                plot_bgcolor='white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("""
                                <div class='stCard'>
                                    <h3 style='text-align: center;'>Churn vs Retain Distribution</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            churn_count = (st.session_state.predictions > 0.5).sum()
                            retain_count = len(st.session_state.predictions) - churn_count
                            fig = px.pie(
                                values=[churn_count, retain_count],
                                names=['Churn', 'Retain'],
                                title="Churn vs Retain Distribution",
                                template='plotly_white',
                                color_discrete_sequence=['#ef5350', '#66bb6a']
                            )
                            fig.update_layout(
                                paper_bgcolor='white',
                                plot_bgcolor='white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    elif page == "Risk Analysis":
                        # Create risk analysis dataframe
                        risk_categories = [st.session_state.model.get_risk_category(p) for p in st.session_state.predictions]
                        risk_df = pd.DataFrame({
                            'Customer ID': df.index + 1,
                            'Churn Probability': [f"{p:.4f}" for p in st.session_state.predictions],
                            'Risk Category': risk_categories
                        })
                        
                        # Convert probability to numeric for sorting
                        risk_df['Probability_Numeric'] = st.session_state.predictions
                        
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>Customer Risk Distribution</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        risk_counts = pd.Series(risk_categories).value_counts()
                        fig = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            title="Customer Risk Distribution",
                            template='plotly_white',
                            color_discrete_sequence=['#ef5350', '#ffa726', '#66bb6a']
                        )
                        fig.update_layout(
                            paper_bgcolor='white',
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk category breakdown
                        col1, col2, col3 = st.columns(3)
                        high_count = (risk_df['Risk Category'] == 'High Risk').sum()
                        medium_count = (risk_df['Risk Category'] == 'Medium Risk').sum()
                        low_count = (risk_df['Risk Category'] == 'Low Risk').sum()
                        
                        with col1:
                            st.metric("High Risk Customers", high_count, delta=None, delta_color="normal")
                        with col2:
                            st.metric("Medium Risk Customers", medium_count, delta=None, delta_color="normal")
                        with col3:
                            st.metric("Low Risk Customers", low_count, delta=None, delta_color="normal")
                        
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>High-Risk Customers</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        high_risk_df = risk_df[risk_df['Risk Category'] == 'High Risk'].sort_values('Probability_Numeric', ascending=False).head(20).drop('Probability_Numeric', axis=1)
                        
                        if len(high_risk_df) > 0:
                            st.dataframe(
                                high_risk_df,
                                use_container_width=True,
                                height=300,
                                hide_index=True
                            )
                        else:
                            st.info("No high-risk customers identified.")
                        
                        # Show all risk categories with search/filter
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>All Customer Risk Analysis</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk category filter
                        risk_filter = st.selectbox(
                            "Filter by Risk Category",
                            ["All", "High Risk", "Medium Risk", "Low Risk"],
                            key="risk_filter"
                        )
                        
                        filtered_df = risk_df.copy()
                        if risk_filter != "All":
                            filtered_df = risk_df[risk_df['Risk Category'] == risk_filter]
                        
                        # Sort options
                        sort_option = st.selectbox(
                            "Sort by",
                            ["Probability (High to Low)", "Probability (Low to High)", "Customer ID"],
                            key="risk_sort"
                        )
                        
                        if sort_option == "Probability (High to Low)":
                            filtered_df = filtered_df.sort_values('Probability_Numeric', ascending=False)
                        elif sort_option == "Probability (Low to High)":
                            filtered_df = filtered_df.sort_values('Probability_Numeric', ascending=True)
                        else:
                            filtered_df = filtered_df.sort_values('Customer ID', ascending=True)
                        
                        # Display filtered results
                        display_df = filtered_df.drop('Probability_Numeric', axis=1).head(50)
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=400,
                            hide_index=True
                        )
                    elif page == "Feature Importance":
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>Feature Importance Analysis</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        importance_df = st.session_state.model.get_feature_importance()
                        
                        # Summary statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Features", len(importance_df))
                        with col2:
                            st.metric("Avg Importance", f"{importance_df['importance'].mean():.4f}")
                        
                        # Feature descriptions dictionary
                        feature_descriptions = {
                            'Total Spend': 'Customer lifetime value - cumulative spending with company',
                            'Tenure': 'Length of relationship - months/years as customer',
                            'Support Calls': 'Frequency of customer service interactions',
                            'Usage Frequency': 'How often customer uses the service',
                            'Payment Delay': 'Days overdue on payments - indicator of dissatisfaction',
                            'Last Interaction': 'Days since last engagement - measures activity',
                            'Subscription Type': 'Service tier level (Basic/Standard/Premium)',
                            'Contract Length': 'Commitment period (Monthly/Quarterly/Annual)',
                            'Spend_Per_Month': 'Average monthly spending - efficiency metric',
                            'Support_Ratio': 'Support calls per month - normalized metric',
                            'Usage_Efficiency': 'Usage frequency relative to tenure',
                            'Subscription_Value': 'Numeric representation of subscription tier',
                            'Spend_vs_Subscription': 'Actual spend vs expected for tier',
                            'Contract_Stability': 'Contract length in months - loyalty indicator'
                        }
                        
                        # Add descriptions to the dataframe
                        importance_df['Description'] = importance_df['feature'].map(
                            lambda x: feature_descriptions.get(x, 'Feature importance for customer churn prediction')
                        )
                        
                        # Display feature importance bar chart with tooltips
                        display_df = importance_df.head(20).copy()
                        fig = px.bar(
                            display_df,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Top 20 Most Important Features",
                            template='plotly_white',
                            color='importance',
                            color_continuous_scale='Viridis',
                            hover_data={'Description': True},
                            hover_name='feature',
                            labels={'importance': 'Importance Score', 'feature': 'Feature Name'}
                        )
                        fig.update_traces(
                            hovertemplate='<b>%{hovertext}</b><br>' +
                                        'Importance: %{x:.4f}<br>' +
                                        '<br>%{customdata[0]}<extra></extra>',
                            customdata=display_df[['Description']]
                        )
                        fig.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            height=600
                        )
                        fig.update_xaxes(title_text="Importance Score")
                        fig.update_yaxes(title_text="Feature Name")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature descriptions accordion
                        st.markdown("#### 📖 Feature Descriptions")
                        st.caption("Hover over features in the chart above to see descriptions, or expand below for a full reference.")
                        
                        with st.expander("📚 Complete Feature Reference Guide", expanded=False):
                            # Group features with descriptions
                            st.markdown("##### High-Value Features")
                            high_importance = display_df.nlargest(5, 'importance')
                            for _, row in high_importance.iterrows():
                                st.markdown(f"""
                                    **{row['feature']}** (Importance: {row['importance']:.4f})
                                    - {row['Description']}
                                """)
                            
                            st.markdown("---")
                            st.markdown("##### Additional Features")
                            for _, row in display_df.iloc[5:].iterrows():
                                st.markdown(f"""
                                    **{row['feature']}** (Importance: {row['importance']:.4f})
                                    - {row['Description']}
                                """)
                        
                        # Full feature importance table with descriptions
                        with st.expander("📊 View All Feature Importances with Descriptions", expanded=False):
                            display_table = importance_df[['feature', 'importance', 'Description']].copy()
                            display_table = display_table.sort_values('importance', ascending=False)
                            st.dataframe(
                                display_table,
                                use_container_width=True,
                                height=400,
                                hide_index=True,
                                column_config={
                                    "feature": st.column_config.TextColumn("Feature Name", width="medium"),
                                    "importance": st.column_config.NumberColumn("Importance", format="%.6f", width="small"),
                                    "Description": st.column_config.TextColumn("Description", width="large")
                                }
                            )
                        
                        st.markdown("""
                            <div class='stCard'>
                                <h3 style='text-align: center;'>SHAP Values Analysis (Test Set)</h3>
                                <p style='text-align: center; color: gray;'>Understanding feature impact on predictions</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        try:
                            shap_values = st.session_state.model.get_shap_values(X_test_processed.iloc[:50])  # Limit to 50 samples for performance
                            
                            # SHAP bar plot
                            with st.spinner('Computing SHAP values...'):
                                fig1, ax1 = plt.subplots(figsize=(10, 8))
                                shap.summary_plot(shap_values, X_test_processed.iloc[:50], plot_type="bar", show=False)
                                st.pyplot(fig1)
                                plt.close(fig1)
                            
                            # SHAP summary plot (beeswarm)
                            st.markdown("""
                                <div class='stCard'>
                                    <h4 style='text-align: center;'>SHAP Summary Plot</h4>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            fig2, ax2 = plt.subplots(figsize=(10, 10))
                            shap.summary_plot(shap_values, X_test_processed.iloc[:50], show=False, max_display=20)
                            st.pyplot(fig2)
                            plt.close(fig2)
                            
                            st.info("💡 **How to read SHAP values:** Red indicates higher feature values push predictions towards churn, while blue indicates lower values. The width shows the impact magnitude.")
                        except Exception as e:
                            st.error(f"Error computing SHAP values: {str(e)}")
                            st.warning("SHAP values may not be available due to dataset size or feature count.")
                    elif page == "Real-time Prediction":
                        st.markdown("""
                            <div style='background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                                        padding: 2rem; 
                                        border-radius: 15px; 
                                        margin-bottom: 2rem;
                                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
                                <h1 style='color: white; text-align: center; margin-bottom: 0.5rem;'>🎯 Real-Time Churn Prediction</h1>
                                <p style='color: rgba(255,255,255,0.9); text-align: center; margin: 0;'>
                                    Enter customer information to get instant churn prediction
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Create input form panel
                        st.markdown("""
                            <div style='background: #f8f9fa; 
                                        padding: 2rem; 
                                        border-radius: 10px; 
                                        border: 2px solid #e0e0e0;
                                        margin-bottom: 2rem;'>
                                <h2 style='text-align: center; color: #333; margin-bottom: 1rem;'>📝 Customer Information</h2>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Store prediction in session state
                        if 'prediction_result' not in st.session_state:
                            st.session_state.prediction_result = None
                        
                        input_data = {}
                        feature_cols = [col for col in df.columns if col != 'churn']
                        
                        # Arrange input fields in columns
                        num_columns = 3
                        num_fields_per_col = (len(feature_cols) + num_columns - 1) // num_columns
                        
                        cols = st.columns(num_columns)
                        for i, col in enumerate(feature_cols):
                            col_idx = i // num_fields_per_col
                            with cols[col_idx]:
                                if df[col].dtype in ['int64', 'float64']:
                                    default_val = float(df[col].mean()) if not pd.isna(df[col].mean()) else 0.0
                                    input_data[col] = st.number_input(
                                        f"**{col}**",
                                        value=default_val,
                                        format="%.2f",
                                        key=f"input_{col}"
                                    )
                                else:
                                    unique_vals = df[col].dropna().unique().tolist()
                                    if unique_vals:
                                        default_val = unique_vals[0]
                                    else:
                                        default_val = ""
                                    input_data[col] = st.selectbox(
                                        f"**{col}**",
                                        options=unique_vals,
                                        key=f"input_{col}"
                                    )
                        
                        # Prediction button with enhanced styling
                        st.markdown("<br>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            predict_button = st.button(
                                "🚀 Get Churn Prediction", 
                                key="predict_button",
                                use_container_width=True,
                                type="primary"
                            )
                        
                        if predict_button:
                            with st.spinner('Analyzing customer data and generating prediction...'):
                                try:
                                    input_df = pd.DataFrame([input_data])
                                    processed_input = st.session_state.data_processor.preprocess_single_row(input_df)
                                    prediction = st.session_state.model.predict(processed_input)[0]
                                    risk_category = st.session_state.model.get_risk_category(prediction)
                                    
                                    # Store in session state
                                    st.session_state.prediction_result = {
                                        'probability': prediction,
                                        'risk_category': risk_category,
                                        'processed_input': processed_input
                                    }
                                except Exception as e:
                                    st.error(f"Error generating prediction: {str(e)}")
                                    st.session_state.prediction_result = None
                        
                        # Display results if prediction exists
                        if st.session_state.prediction_result is not None:
                            prediction = st.session_state.prediction_result['probability']
                            risk_category = st.session_state.prediction_result['risk_category']
                            processed_input = st.session_state.prediction_result['processed_input']
                            
                            # Results panel with gradient background
                            risk_colors = {
                                'High Risk': '#ef5350',
                                'Medium Risk': '#ffa726',
                                'Low Risk': '#66bb6a'
                            }
                            
                            st.markdown("""
                                <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                            padding: 2rem; 
                                            border-radius: 15px; 
                                            margin: 2rem 0;
                                            box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
                                    <h2 style='color: white; text-align: center; margin-bottom: 1rem;'>✨ Prediction Result</h2>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Results metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                    <div style='background: white; 
                                                padding: 1.5rem; 
                                                border-radius: 10px; 
                                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                                                text-align: center;'>
                                        <h4 style='color: #666; margin: 0 0 0.5rem 0;'>Churn Probability</h4>
                                        <h1 style='color: #1e3c72; margin: 0;'>{prediction:.2%}</h1>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                risk_color = risk_colors[risk_category]
                                st.markdown(f"""
                                    <div style='background: white; 
                                                padding: 1.5rem; 
                                                border-radius: 10px; 
                                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                                                text-align: center;'>
                                        <h4 style='color: #666; margin: 0 0 0.5rem 0;'>Risk Category</h4>
                                        <h1 style='color: {risk_color}; margin: 0;'>{risk_category}</h1>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                # Display recommendation based on risk
                                if risk_category == 'High Risk':
                                    rec = "Immediate Action Needed"
                                    rec_color = "#ef5350"
                                elif risk_category == 'Medium Risk':
                                    rec = "Monitor Closely"
                                    rec_color = "#ffa726"
                                else:
                                    rec = "Low Priority"
                                    rec_color = "#66bb6a"
                                
                                st.markdown(f"""
                                    <div style='background: white; 
                                                padding: 1.5rem; 
                                                border-radius: 10px; 
                                                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                                                text-align: center;'>
                                        <h4 style='color: #666; margin: 0 0 0.5rem 0;'>Recommendation</h4>
                                        <h3 style='color: {rec_color}; margin: 0;'>{rec}</h3>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Feature Impact Analysis
                            st.markdown("""
                                <div style='background: #f8f9fa; 
                                            padding: 1.5rem; 
                                            border-radius: 10px; 
                                            margin-top: 2rem;
                                            border: 2px solid #e0e0e0;'>
                                    <h3 style='text-align: center; color: #333; margin-bottom: 1rem;'>🔍 Feature Impact Analysis</h3>
                                    <p style='text-align: center; color: #666;'>
                                        Understanding which features drive this prediction
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            try:
                                shap_values = st.session_state.model.get_shap_values(processed_input)
                                fig, ax = plt.subplots(figsize=(10, 6))
                                # Handle both single and multi-dimensional SHAP values
                                if len(shap_values.shape) > 1:
                                    shap_values_to_plot = shap_values[0]
                                else:
                                    shap_values_to_plot = shap_values
                                shap.bar_plot(shap_values_to_plot, processed_input, show=False)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception as e:
                                st.warning(f"Could not generate SHAP plot: {str(e)}")
                            
                            st.markdown("---")
                    # Download predictions with enhanced styling and visual separation
                    st.markdown("---")
                    st.markdown("""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 2rem; 
                                    border-radius: 15px; 
                                    margin: 2rem 0;
                                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
                            <h2 style='color: white; text-align: center; margin-bottom: 1rem;'>📊 Export Predictions</h2>
                            <p style='color: rgba(255,255,255,0.9); text-align: center; margin-bottom: 0;'>
                                Download all customer churn predictions for further analysis
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    predictions_df = pd.DataFrame({
                        'Customer ID': df.index + 1,
                        'Churn Probability': [f"{p:.6f}" for p in st.session_state.predictions],
                        'Risk Category': [st.session_state.model.get_risk_category(p) for p in st.session_state.predictions]
                    })
                    
                    # Enhanced download button with info
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # File info
                        total_rows = len(predictions_df)
                        st.info(f"📈 **Ready to export:** {total_rows} customer predictions")
                        
                        # Download button with custom styling
                        csv_data = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Predictions CSV",
                            data=csv_data,
                            file_name=f"churn_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_button",
                            use_container_width=True,
                            type="primary"
                        )
                        
                        st.caption("💾 CSV includes: Customer ID, Churn Probability, and Risk Category")
                        st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Optional: Show sample of data being downloaded
                    with st.expander("👀 Preview Data to be Downloaded", expanded=False):
                        st.dataframe(
                            predictions_df.head(20),
                            use_container_width=True,
                            height=300,
                            hide_index=True
                        )
                        if len(predictions_df) > 20:
                            st.caption(f"Showing first 20 of {len(predictions_df)} rows...")
                    
                    st.markdown("---")
            else:
                st.error("❌ The uploaded CSV does not contain a 'churn' column. Please include the target column for training.")
                st.info("💡 Your CSV file should contain a 'churn' column with 0/1 or Yes/No values.")
    except pd.errors.EmptyDataError:
        st.error("❌ The CSV file is empty or corrupted.")
        st.warning("Please check your CSV file and ensure it contains valid data.")
    except pd.errors.ParserError as e:
        st.error(f"❌ Error parsing CSV file: {str(e)}")
        st.warning("Please check your CSV file format. Ensure it's properly formatted with comma-separated values.")
        st.info("💡 Common issues: Missing quotes around values with commas, encoding problems, or malformed rows.")
    except UnicodeDecodeError as e:
        st.error(f"❌ Encoding error: Unable to read the file with default encoding.")
        st.warning("Please save your CSV file with UTF-8 encoding and try again.")
        st.info("💡 Most spreadsheet programs (Excel, Google Sheets) can export CSV files with UTF-8 encoding.")
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.warning("An unexpected error occurred. Please check your file and try again.")
        
        # Show detailed error in expander for debugging
        with st.expander("🔍 Show detailed error information"):
            import traceback
            st.code(traceback.format_exc())
        
        st.info("💡 If the problem persists, ensure your CSV file is properly formatted and contains the required columns.")
        st.stop() 