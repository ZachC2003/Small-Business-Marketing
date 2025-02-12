import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import numpy as np
import feedparser
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import AsyncOpenAI  # Updated import statement
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI  # Updated import statement
from langchain_openai import OpenAIEmbeddings  # Import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore # Import InMemoryVectorStore
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

# Ensure the unstructured package is installed
try:
    from unstructured.__version__ import __version__ as __unstructured_version__
except ModuleNotFoundError:
    import subprocess
    subprocess.check_call(["pip", "install", "unstructured"])

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment variable
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # Updated client initialization

def load_and_process_excel(file_path):
    df = pd.read_excel(file_path)
    embeddings = OpenAIEmbeddings()  # Specify the embedding model
    
    # Create the VectorstoreIndexCreator with the embedding model
    index_creator = VectorstoreIndexCreator(
        embedding=embeddings,
        vectorstore_cls=InMemoryVectorStore
    )
    
    # Convert DataFrame to a list of Documents
    documents = [Document(page_content=row.to_json(), metadata={"index": index}) for index, row in df.iterrows()]
    
    # Use the index_creator to process your documents
    index = index_creator.from_documents(documents)
    
    if 'initial_data' not in st.session_state:
        st.session_state.initial_data = index
    
    return index

def load_software_database():
    return {
        'Diesel Repair': {
            'Marketing': ['HubSpot', 'Mailchimp', 'Constant Contact'],
            'CRM': ['Shopmonkey', 'Tekmetric', 'Shop-Ware'],
            'Accounting': ['QuickBooks', 'Xero', 'FreshBooks']
        },
        'Landscaping': {
            'Marketing': ['HubSpot', 'Mailchimp', 'Constant Contact'],
            'CRM': ['Service Autopilot', 'Jobber', 'Lawn Guru'],
            'Accounting': ['QuickBooks', 'FreshBooks', 'Wave']
        }
    }

def identify_weakness(data):
    return data[data['Rating %'] == data['Rating %'].min()]['Category'].iloc[0]

def extract_software_from_question(question):
    software_keywords = {
        'crm': ['Shopmonkey', 'Tekmetric', 'Service Autopilot', 'Jobber'],
        'accounting': ['QuickBooks', 'Xero', 'FreshBooks', 'Wave'],
        'marketing': ['HubSpot', 'Mailchimp', 'Constant Contact']
    }
    
    for category, software_list in software_keywords.items():
        if category in question.lower():
            return software_list[0]
    return 'Generic Software'

def extract_business_metrics(data):
    return {
        'total_score': data['Total'].mean() if 'Total' in data else 0,
        'rating_percentage': data['Rating %'].mean() if 'Rating %' in data else 0
    }

def estimate_savings(software, metrics):
    base_savings = {
        'Shopmonkey': 1200,
        'QuickBooks': 800,
        'HubSpot': 1500
    }
    return base_savings.get(software, 1000) * metrics['rating_percentage']

def fetch_implementation_steps(software):
    implementation_guides = {
        'Shopmonkey': [
            'Sign up for a free trial',
            'Import your customer database',
            'Set up service templates',
            'Configure user permissions',
            'Train staff on basic features'
        ],
        'QuickBooks': [
            'Create your account',
            'Set up your company profile',
            'Connect your bank accounts',
            'Import customer and vendor lists',
            'Configure chart of accounts'
        ]
    }
    return implementation_guides.get(software, ['No specific implementation guide available'])

def format_as_numbered_list(steps):
    return '\n'.join(f"{i+1}. {step}" for i, step in enumerate(steps))

def suggest_improvements(metrics_data, business_type):
    low_scores = metrics_data[metrics_data['Score (0-10)'] < 5]
    software_db = load_software_database()
    
    recommendations = {}
    for _, row in low_scores.iterrows():
        category = row['Category']
        if category in software_db.get(business_type, {}):
            recommendations[category] = software_db[business_type][category]
    
    return recommendations

def load_data_fixed_v5(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name="Current Analysis")
    
    # Process Metrics Data
    metrics_data = df[['Category', 'Subcategory', 'Metric', 'Metric Score (0-10)']].dropna(subset=['Metric Score (0-10)'])
    metrics_data = metrics_data.rename(columns={'Metric Score (0-10)': 'Score (0-10)'})
    
    # Extract Subcategory Data
    sub_categories = df[['Sub Category', 'Category Score', 'Benchmark', 'Rating %']].dropna(subset=['Rating %'])
    sub_categories = sub_categories.rename(columns={
        'Sub Category': 'Subcategory',
        'Category Score': 'Total'
    })

    # Extract Main Category Data
    main_categories = df[['Main Category', 'Category Score.1', 'Benchmark.1', 'Rating %.1']].dropna(subset=['Rating %.1'])
    main_categories = main_categories.rename(columns={
        'Main Category': 'Category',
        'Category Score.1': 'Total',
        'Benchmark.1': 'Benchmark',
        'Rating %.1': 'Rating %'
    })
    
    # Extract Whole Report Data
    whole_report = df[['Whole Report', 'Value']].dropna(subset=['Value'])
    whole_report = whole_report.set_index('Whole Report')['Value']
    
    return metrics_data, sub_categories, main_categories, whole_report

def create_overall_radar_chart(data):
    # Select rows where 'Category' is not empty (Assuming these are main categories)
    main_categories = data.dropna(subset=['Category']).drop_duplicates(subset=['Category'])

    categories = main_categories['Category'].tolist()
    values = main_categories['Rating %'].tolist()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Overall Performance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title='Overall Category Performance'
    )
    
    return fig

def create_subcategory_radar_chart(data):
    categories = data['Category'].unique()
    
    fig = go.Figure()
    
    for category in categories:
        category_data = data[data['Category'] == category]
        fig.add_trace(go.Scatterpolar(
            r=[category_data['Rating %'].values[0]],
            theta=[category],
            fill='toself',
            name=category
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title='Category Performance'
    )
    
    return fig

def create_category_comparison_charts(data, main_categories):
    # Calculate category scores using Rating % instead of Score (0-10)
    category_scores = data[data['Category'].isin(main_categories)].groupby('Category')['Rating %'].mean()
    
    # Create bar chart with percentage values
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=category_scores.index,
        y=category_scores.values * 100,  # Convert to percentage
        marker_color=['red' if x < 0.5 else 'yellow' if x < 0.75 else 'green' for x in category_scores.values]
    ))
    bar_fig.update_layout(
        title="Category Performance Comparison",
        yaxis_title="Rating %",
        height=400
    )
    
    # Create pie chart with percentage values
    pie_fig = go.Figure(data=[go.Pie(
        labels=category_scores.index,
        values=category_scores.values * 100,  # Convert to percentage
        marker=dict(
            colors=['red' if x < 0.5 else 'yellow' if x < 0.75 else 'green' for x in category_scores.values]
        )
    )])
    pie_fig.update_layout(
        title="Category Distribution",
        height=400
    )
    
    return bar_fig, pie_fig

# Function to generate AI strategy report
def fetch_blog_posts(rss_url):
    feed = feedparser.parse(rss_url)
    return [entry.summary for entry in feed.entries]

def fetch_software_reviews(api_url, api_key):
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(api_url, headers=headers)
    return response.json()

def match_needs_with_software(company_needs, software_data):
    vectorizer = TfidfVectorizer()
    needs_vector = vectorizer.fit_transform([company_needs])
    software_vectors = vectorizer.transform(software_data)
    similarities = cosine_similarity(needs_vector, software_vectors)
    return similarities.argsort()[0][-3:][::-1]  # Return top 3 matches

async def generate_financial_analysis(metrics_data, business_type, analysis_option):
    prompt = f"As a financial advisor for a {business_type} company, provide a detailed analysis on {analysis_option}. Use the following metrics data: {metrics_data.to_string()}"
    
    response = await client.chat.completions.create(
        model="gpt-4",  # Updated model
        messages=[
            {"role": "system", "content": "You are an expert financial advisor."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()  # Corrected way to access response content

async def generate_strategy_report(data):
    # Identify company needs based on low scores
    low_scoring_areas = data[data['Rating %'] < 0.5]['Category'].tolist()
    company_needs = " ".join(str(area) for area in low_scoring_areas)
    
    # Generate report using OpenAI
    prompt = f"Based on the following marketing performance data, provide a concise strategy report:\n\n{data.to_string()}\n\nStrategy Report:"
    
    response = await client.chat.completions.create(
        model="gpt-4",  # Updated model
        messages=[
            {"role": "system", "content": "You are a marketing strategy expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    
    initial_report = response.choices[0].message.content.strip()  # Corrected way to access response content
    final_report = f"{initial_report}\n\nBased on the analysis, we recommend focusing on improving: {', '.join(low_scoring_areas)}."
    
    return final_report

async def analyze_market_trends(query, business_type):
    prompt = f"Analyze the latest market trends and competitive landscape for a {business_type} business, focusing on: {query}"
    response = await client.chat.completions.create(
        model="gpt-4",  # Updated model
        messages=[
            {"role": "system", "content": "You are a market research expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()  # Corrected way to access response content

async def generate_chat_response(data, user_question):
    chat_prompt = f"Based on the following marketing performance data:\n\n{data.to_string()}\n\nUser question: {user_question}\n\nResponse:"
    chat_response = await client.chat.completions.create(
        model="gpt-4",  # Updated model
        messages=[
            {"role": "system", "content": "You are a marketing strategy expert."},
            {"role": "user", "content": chat_prompt}
        ],
        max_tokens=150
    )
    return chat_response.choices[0].message.content.strip()  # Corrected way to access response content

async def advanced_ai_assistant(query, metrics_data):
    prompt = f"""You are an AI assistant for small businesses. 
    Based on the following metrics data: {metrics_data.to_string()}, 
    and the user query: "{query}", 
    provide a detailed response including:
    1. Specific recommendations for improvement
    2. Suggested strategies or tools
    3. Implementation steps
    4. Explanation of the importance and potential impact
    Keep the response concise but informative."""

    response = await client.chat.completions.create(
        model="gpt-4",  # Updated model
        messages=[
            {"role": "system", "content": "You are a knowledgeable small business consultant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()  # Corrected way to access response content

def generate_implementation_guide(software):
    steps = fetch_implementation_steps(software)
    return format_as_numbered_list(steps)

def main_categories_page(main_categories, whole_report, metrics_data):
    st.header("Main Categories Overview")
    
    st.subheader("Overall Main Categories Performance")
    # Overall Performance - using whole_report
    overall_score = whole_report['Total Rating %'] * 100
    st.metric("Overall Performance", f"{overall_score:.1f}%")
    main_categories_list = ['Business Profiles', 'Social Media Presence', 'Website Performance', 'Advertising Efforts', 'Brand Consistency']
    
    # Bar Chart for Category Ratings - using main_categories
    fig = go.Figure()
    filtered_data = main_categories[main_categories['Category'].isin(main_categories_list)]
    fig.add_trace(go.Bar(
        x=filtered_data['Category'],
        y=filtered_data['Rating %'] * 100,
        marker=dict(color=['green' if x >= 0.75 else 'yellow' if x >= 0.5 else 'red' for x in filtered_data['Rating %']])
    ))
    fig.update_layout(title="Category Performance Ratings", yaxis_title="Rating %")
    st.plotly_chart(fig)
    
    # Individual Category Performance - using main_categories
    st.subheader("Individual Category Breakdowns")
    for category in main_categories_list:
        category_data = main_categories[main_categories['Category'] == category]
        if not category_data.empty:
            row = category_data.iloc[0]
            st.write(f"**{category}**")
            st.write(f"Total Score: {row['Total']}")
            st.write(f"Benchmark: {row['Benchmark']}")
            st.write(f"Rating: {row['Rating %']*100:.1f}%")
            
            # Gauge Chart
            fig = go.Figure()
            fig.add_trace(go.Pie(values=[row['Rating %'], 1 - row['Rating %']],
                                 hole=0.7,
                                 marker_colors=['green', 'red'],
                                 textinfo='none'))
            fig.update_layout(title=f"{category} Performance", showlegend=False)
            st.plotly_chart(fig)
            st.write("---")

            st.write("**Individual Metrics:**")
            category_metrics = metrics_data[metrics_data['Category'] == category]
            for _, metric in category_metrics.iterrows():
                st.write(f"{metric['Metric']}: {metric['Score (0-10)']}")
            st.write("---")

def subcategories_page(sub_categories, metrics_data):
    st.header("Subcategories Breakdown")
    
    st.subheader("Overall Subcategory Performance")
    
    # Bar chart for subcategories
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sub_categories['Subcategory'],
        y=sub_categories['Rating %'] * 100,
        marker=dict(color=['green' if x >= 0.75 else 'yellow' if x >= 0.5 else 'red' for x in sub_categories['Rating %']])
    ))
    fig.update_layout(title="Subcategory Performance Ratings", yaxis_title="Rating %")
    st.plotly_chart(fig)
    
    # Individual Subcategory Performance
    st.subheader("Individual Subcategory Breakdowns")
    for _, row in sub_categories.iterrows():
        st.write(f"**{row['Subcategory']}**")
        st.write(f"Total Score: {row['Total']}")
        st.write(f"Benchmark: {row['Benchmark']}")
        st.write(f"Rating: {row['Rating %']*100:.1f}%")
        
        # Gauge Chart
        fig = go.Figure()
        fig.add_trace(go.Pie(values=[row['Rating %'], 1 - row['Rating %']],
                             hole=0.7,
                             marker_colors=['green', 'red'],
                             textinfo='none'))
        fig.update_layout(title=f"{row['Subcategory']} Performance", showlegend=False)
        st.plotly_chart(fig)
        st.write("---")

        st.write("**Individual Metrics:**")
        subcategory_metrics = metrics_data[metrics_data['Subcategory'] == row['Subcategory']]
        for _, metric in subcategory_metrics.iterrows():
            st.write(f"{metric['Metric']}: {metric['Score (0-10)']}")
        st.write("---")

async def statistics_strategy_page(metrics_data, sub_categories, main_categories, whole_report):
    st.header("Statistics and Strategy Report")
    
    # Overall Statistics from whole report
    st.subheader("Overall Statistics")
    
    try:
        whole_report_stats = pd.DataFrame({
            'Total Score': [whole_report['Total Score']],
            'Total Benchmark': [whole_report['Total Benchmark']],
            'Total Rating %': [whole_report['Total Rating %']]
    }, index=['Overall'])


        # Display with proper formatting
        st.dataframe(whole_report_stats.style.format({
            'Total Score': '{:.0f}',
            'Total Benchmark': '{:.0f}',
            'Total Rating %': '{:.2%}'
        }))
    except KeyError as e:
        st.error(f"Error: Unable to find key {e} in whole_report. Please check the data structure.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
    
    # Top and Bottom Performers - Main Categories
    st.subheader("Main Categories Performance")
    if 'Category' in main_categories.columns:
        main_category_scores = main_categories.sort_values('Rating %', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top 2 Best Main Categories**")
            st.write(main_category_scores[['Category', 'Rating %']].head(2).style.format({'Rating %': '{:.1%}'}))
        
        with col2:
            st.write("**Top 2 Worst Main Categories**")
            st.write(main_category_scores[['Category', 'Rating %']].tail(2).style.format({'Rating %': '{:.1%}'}))
                                                                                    
    
    # Top and Bottom Performers - Subcategories
    st.subheader("Subcategories Performance")
    sub_category_scores = sub_categories.sort_values('Rating %', ascending=False)
    
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Top 2 Best Subcategories**")
        st.write(sub_category_scores[['Subcategory', 'Rating %']].head(2).style.format({'Rating %': '{:.1%}'}))
    
    with col4:
        st.write("**Top 2 Worst Subcategories**")
        st.write(sub_category_scores[['Subcategory', 'Rating %']].tail(2).style.format({'Rating %': '{:.1%}'}))

    # Advanced AI Business Assistant
    st.subheader("Advanced AI Business Assistant")
    user_query = st.text_area("Ask a detailed question about improving your business:")
    if user_query:
        with st.spinner("Generating comprehensive response..."):
            ai_response = await advanced_ai_assistant(user_query, pd.concat([sub_categories, main_categories]))  # Await the coroutine
        st.write(ai_response)

def langchain_analysis_page(index):
    st.header("LangChain Analysis")

    if 'initial_data' in st.session_state:
        st.subheader("Initial Data vs Current Data")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Initial Data")
            display_data_summary(st.session_state.initial_data)
        
        with col2:
            st.write("Current Data")
            display_data_summary(index)
        
        st.write("Improvements")
        display_improvements(st.session_state.initial_data, index)

    query = st.text_input("Ask a question about your data:")
    if query:
        llm = OpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=index.vectorstore.as_retriever())
        response = qa_chain.run(query)
        st.write(response)

    st.subheader("Data Insights")
    if st.button("Generate Insights"):
        insights = generate_insights(index)
        st.write(insights)

def generate_insights(index):
    llm = OpenAI(temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=index.vectorstore.as_retriever())
    insights_prompt = "Analyze the marketing data and provide three key insights or recommendations for improving the business."
    insights = qa_chain.run(insights_prompt)
    return insights

def display_data_summary(index):
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=index.vectorstore.as_retriever())
    summary = qa_chain.run("Provide a brief summary of the marketing performance data.")
    st.write(summary)

def display_improvements(initial_index, current_index):
    llm = OpenAI(temperature=0)
    initial_qa = RetrievalQA.from_chain_type(llm, retriever=initial_index.vectorstore.as_retriever())
    current_qa = RetrievalQA.from_chain_type(llm, retriever=current_index.vectorstore.as_retriever())
    
    initial_performance = initial_qa.run("What is the overall marketing performance?")
    current_performance = current_qa.run("What is the overall marketing performance?")
    
    prompt = f"Compare the following marketing performances and list the improvements:\n\nInitial: {initial_performance}\n\nCurrent: {current_performance}"
    
    # Pass the prompt as a list of strings
    improvements = llm.generate([prompt])
    
    # Extract the generated text from the result
    improvement_text = improvements.generations[0][0].text
    
    st.write(improvement_text)

async def advanced_ai_assistant(query, metrics_data):
    prompt = f"""You are an AI assistant for small businesses. 
    Based on the following metrics data: {metrics_data.to_string()}, 
    and the user query: "{query}", 
    provide a detailed response including:
    1. Specific recommendations for improvement
    2. Suggested strategies or tools
    3. Implementation steps
    4. Explanation of the importance and potential impact
    Keep the response concise but informative."""

    response = await client.chat.completions.create(
        model="gpt-4",  # Updated model
        messages=[
            {"role": "system", "content": "You are a knowledgeable small business consultant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()  # Corrected way to access response content

async def main():
    st.set_page_config(page_title="Small Business Optimization Platform", layout="wide")
    st.title("Small Business Optimization Platform")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Excel file", 
        type=["xlsx"],
        accept_multiple_files=False,
        help="Upload your marketing analysis Excel file with 'Current Analysis' sheet"
    )

    if uploaded_file:
        metrics_data, sub_categories, main_categories, whole_report = load_data_fixed_v5(uploaded_file)
        
        # Process Excel file with LangChain
        index = load_and_process_excel(uploaded_file)
        
        page = st.sidebar.radio("Navigate to", ["Main Categories", "Subcategories", "Statistics & Strategy", "LangChain Analysis"])
        
        if page == "Main Categories":
            main_categories_page(main_categories, whole_report, metrics_data)
        elif page == "Subcategories":
            subcategories_page(sub_categories, metrics_data)
        elif page == "Statistics & Strategy":
            await statistics_strategy_page(metrics_data, sub_categories, main_categories, whole_report)  # Await the coroutine
        elif page == "LangChain Analysis":
            langchain_analysis_page(index)
    else:
        st.info("Please upload an Excel file to begin.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())  # Run the async main function
