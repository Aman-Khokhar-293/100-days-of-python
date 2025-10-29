# Interactive Streamlit dashboard for AutoML

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ai_automl_engine import AutoMLEngine
import io


def main():
    st.set_page_config(
        page_title="AI AutoML Dashboard",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Day 100 - AI-Powered Data Science Workflow Automator")
    st.markdown("**Automated ML Pipeline with Intelligent Model Selection**")
    st.divider()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        task_type = st.selectbox(
            "Task Type",
            ["auto", "classification", "regression"],
            help="Select 'auto' for automatic detection"
        )
        
        st.divider()
        st.header("📤 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format"
        )
        
        st.divider()
        st.header("📊 Sample Datasets")
        use_sample = st.checkbox("Use sample dataset")
        if use_sample:
            sample_choice = st.radio(
                "Choose dataset",
                ["Iris (Classification)", "Diabetes (Regression)"]
            )
    
    if uploaded_file is not None or (use_sample if 'use_sample' in locals() else False):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} rows")
        else:
            if sample_choice == "Iris (Classification)":
                from sklearn.datasets import load_iris
                iris = load_iris()
                df = pd.DataFrame(iris.data, columns=iris.feature_names)
                df['target'] = iris.target
            else:
                from sklearn.datasets import load_diabetes
                diabetes = load_diabetes()
                df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
                df['target'] = diabetes.target
            st.info(f"📊 Using sample dataset: {sample_choice}")
        
        with st.expander("📋 Data Preview", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", len(df))
            col2.metric("Columns", len(df.columns))
            col3.metric("Missing Values", df.isnull().sum().sum())
            
            st.dataframe(df.head(), width='stretch')
        
        with st.expander("📈 Data Statistics"):
            st.dataframe(df.describe(), width='stretch')
        
        st.divider()
        st.subheader("🎯 Select Target Column")
        target_column = st.selectbox(
            "Target variable",
            df.columns.tolist(),
            index=len(df.columns) - 1
        )
        
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            with st.spinner("Training models..."):
                automl = AutoMLEngine(task_type=task_type)
                
                if 'automl' not in st.session_state:
                    st.session_state.automl = automl
                
                X, y = automl.preprocess_data(df, target_column)
                X_test, y_test = automl.train_models(X, y)
                
                st.session_state.trained = True
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
            st.success("✅ Training completed!")
            st.balloons()
        
        if st.session_state.get('trained', False):
            st.divider()
            st.header("📊 Results")
            
            automl = st.session_state.automl
            summary = automl.get_model_summary()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Best Model")
                st.metric("Model", summary['best_model'])
                metric_name = "Accuracy" if summary['task_type'] == 'classification' else "R² Score"
                st.metric(metric_name, f"{summary['best_score']:.4f}")
            
            with col2:
                st.subheader("📈 All Models Performance")
                results_data = []
                for name, result in summary['all_results'].items():
                    if summary['task_type'] == 'classification':
                        results_data.append({'Model': name, 'Accuracy': result['accuracy']})
                    else:
                        results_data.append({
                            'Model': name,
                            'R² Score': result['r2_score'],
                            'MSE': result['mse']
                        })
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, width='stretch')
            
            if summary['feature_importance'] is not None:
                st.divider()
                st.subheader("🎯 Feature Importance")
                
                fig = px.bar(
                    summary['feature_importance'].head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 10 Most Important Features',
                    color='importance',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("📊 Model Comparison")
            
            if summary['task_type'] == 'classification':
                scores = [result['accuracy'] for result in summary['all_results'].values()]
            else:
                scores = [result['r2_score'] for result in summary['all_results'].values()]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(summary['all_results'].keys()),
                    y=scores,
                    marker_color=['gold' if name == summary['best_model'] 
                                 else 'lightblue' 
                                 for name in summary['all_results'].keys()]
                )
            ])
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis_title=metric_name,
                height=400
            )
            st.plotly_chart(fig, width='stretch')
            
            st.divider()
            st.subheader("📥 Export Results")
            report = f"""
AI AutoML Report
================

Task Type: {summary['task_type'].upper()}
Best Model: {summary['best_model']}
Best Score: {summary['best_score']:.4f}

Model Results:
--------------
"""
            for name, result in summary['all_results'].items():
                report += f"\n{name}:\n"
                if summary['task_type'] == 'classification':
                    report += f"  Accuracy: {result['accuracy']:.4f}\n"
                else:
                    report += f"  R² Score: {result['r2_score']:.4f}\n"
                    report += f"  MSE: {result['mse']:.4f}\n"
            
            if summary['feature_importance'] is not None:
                report += "\n\nTop Feature Importance:\n"
                report += summary['feature_importance'].head(10).to_string()
            
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name="automl_report.txt",
                mime="text/plain"
            )
    
    else:
        st.info("👈 Please upload a dataset or select a sample dataset from the sidebar to get started")
        
        st.markdown("""
        ### 🎯 Features
        - **Automatic Task Detection**: Automatically identifies classification vs regression
        - **Multiple Models**: Trains and compares multiple algorithms
        - **Feature Engineering**: Intelligent preprocessing and encoding
        - **Visual Analytics**: Interactive charts and model comparisons
        - **Export Results**: Download comprehensive reports
        
        ### 🚀 How to Use
        1. Upload your CSV file or select a sample dataset
        2. Choose your target column
        3. Click "Train Models" and let AI do the work!
        4. Explore results and download reports
        """)


if __name__ == "__main__":
    main()
