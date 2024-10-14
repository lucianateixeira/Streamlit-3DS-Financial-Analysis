<h1>Capstone Data Analytics Dashboard</h1>

<p>
  The <strong>Capstone Data Analytics Dashboard</strong> is a web application built with <strong>Streamlit</strong> that provides users with interactive data visualizations and insights. The application has been deployed and is available at the following URL:
</p>

<p>🌐 <a href="https://capstone-dataanalytics.streamlit.app">Live App</a></p>

<h2>How It Works</h2>

<p>
  The dashboard allows users to explore datasets and customize the visualizations through an intuitive interface. The following features are supported:
</p>

<ul>
  <li><strong>Interactive Filters</strong>: Modify the displayed data based on the chosen filters and selections.</li>
  <li><strong>Customizable Visualizations</strong>: Choose different charts and graphs to visualize the data.</li>
  <li><strong>Real-Time Data Updates</strong>: Data visualizations update as the filters are applied.</li>
</ul>

<h3>Run the App Locally</h3>

<p>To run the application locally, follow these steps:</p>

<pre><code>
# Clone the repository
git clone https://github.com/your-username/capstone-dataanalytics.git

# Navigate into the project directory
cd capstone-dataanalytics

# Install required dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
</code></pre>

<p>After running the app, it will be accessible at <strong>http://localhost:8501/</strong>.</p>

<h2>Example Code Snippet</h2>

<p>Here’s an example of how a section of the dashboard’s code might look:</p>

<pre><code>
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data.csv')

# Sidebar filters
option = st.sidebar.selectbox('Select option:', data['column'].unique())

# Filter data
filtered_data = data[data['column'] == option]

# Display chart
st.line_chart(filtered_data)
</code></pre>

<p>
  The app can be easily adapted to various datasets. Simply place your dataset in the <strong>data/</strong> folder, and adjust the code accordingly to load and display it.
</p>

<h2>Questions</h2>

<h3>What happens after running the app?</h3>
<p>The app will launch in your browser, where you can explore different features and datasets.</p>

<h3>How do I add new features?</h3>
<p>You can extend the app by adding more filters, charts, and analytics by modifying the code in <strong>app.py</strong>.</p>
