# Import python packages
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import datetime 
import uuid
import altair as alt
import re
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
from collections import defaultdict
from collections import Counter
from matplotlib.colors import Normalize
from streamlit_echarts import st_echarts
import math
import ast
from streamlit_extras.app_logo import add_logo

#def get_active_session():
#    if "snowpark_session" not in st.session_state:
#        session = Session.builder.configs(json.load(open("connection.json"))).create()
#        st.session_state['snowpark_session'] = session
#    else:
#        session = st.session_state['snowpark_session']
#    return session

# Call function to create new or get existing Snowpark session to connect to Snowflake
session = get_active_session()

st.set_page_config(layout="wide")

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)
   
#--------------------------------------
#HOME PAGE
#--------------------------------------
def home():
        st.write("")
        st.write("")
        st.image("snowpathimage.png",use_container_width=True)
        #st.image("https://i.postimg.cc/8PWc0cMf/snowpathlogo2.png",use_container_width=True)
    
        with st.expander("**ABOUT**"):
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
             <b>Snowpath</b><sup>™</sup> native application allows users to easily and visually perform and deep dive into <span style="color:#29B5E8;">Path Analysis</span>, <span style="color:#29B5E8;">Attribution Analysis</span> and <span style="color:#29B5E8;">Association Analysis</span> by simply specifying a few parameters in drop-down menus. Leveraging advanced techniques, <b>Snowpath</b><sup>™</sup> intuitively and visually helps identify touchpoints influencing customer (or machine) behaviours, targets them to create segments, performs cross-population behavioural comparisons, computes rule-based and ML-driven attribution models to understand the contribution of each event preceding a specific outcome, and conducts association analysis to uncover hidden patterns and relationships between events. <b>Snowpath</b><sup>™</sup> also harnesses the interpretive and generative power of LLMs thanks to Snowflake AISQL to explain journeys, attribution models, association rules and derive insights (summarize and analyze results, describe behaviors and even suggest actions !)
        </h5>
        """, unsafe_allow_html=True)
            
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
             Visualizing and identifying paths can itself be actionable and often uncovers an area of interest for additional analysis. First, the picture revealed by path analysis can be further enriched with attribution and association analysis. Attribution helps quantify the contribution of individual touchpoints to a defined outcome, while association analysis uncovers relationships between events that frequently occur together. Together, these techniques provide a holistic understanding of event sequences, enabling data-driven decision-making and uncovering new opportunities for optimization. Second, path insights can be used directly to predict outcomes (<span style="color:#29B5E8;">Predictive Modeling</span>) or to derive behavioral features (such as the frequency of specific patterns). These features can then be integrated into existing predictive models, enhancing their accuracy and enabling deeper behavioral segmentation.
        </h5>
        """, unsafe_allow_html=True)
            
        st.markdown("""
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                */background-color: #f0f2f6;*/
                color: #666;
                padding: 10px;
                font-size: 12px;
                text-align: center;
                z-index: 999;
                border-top: 1px solid #ccc;
            }
        </style>
        <div class="footer">
            Copyright &copy; 2025 Yannis Marigo. All rights reserved.
        </div>
        """, unsafe_allow_html=True)

#--------------------------------------
#HELP PAGE
#--------------------------------------
def help():
            #Page Title
        st.markdown("""
        <style>
        .custom-container-1 {
            padding: 10px 10px 10px 10px;
            border-radius: 10px;
            background-color: #f0f2f6;  /* Light blue background */
            border: 1px solid #29B5E8;  /* Blue border */
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-container-1">
            <h5 style="font-size: 18px; font-weight: normal; color: #29B5E8; margin-top: 0px; margin-bottom: -15px;">
                USER GUIDE
            </h5>
        </div>
        """, unsafe_allow_html=True)
    
        with st.expander("**Path Analysis**"):
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            Path analysis is a statistical technique that reveals the sequences and patterns customers follow as they interact with your company or your brand. It helps uncover the specific paths customers follow as they move through various touchpoints, highlighting the most influential events driving customer behavior, their sequencing, and the underlying factors shaping their decision-making process.
        </h5>
        """, unsafe_allow_html=True)
            
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            By analyzing the paths customers take, businesses can identify common behaviors, preferences, and pain points. This information can then be used to tailor marketing strategies, personalize customer interactions, build path-based behavioral segmentations, and ultimately enhance the overall customer experience.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
             Path analysis can increasingly be applied beyond traditional customer journeys to other fields, such as IoT, manufacturing, and complex operational or industrial workflows.
        </h5>
        """, unsafe_allow_html=True)
            

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
             <b>Snowpath</b><sup>™</sup> helps identifying paths and patterns in data in a valuable way to gain insight into the occurrences leading to or from any event of interest or between two events of interest. In the <span style="color:#29B5E8;">Analyze</span> tab of the <span style="color:#29B5E8;">Path Analysis</span> page, users can visually dive deep into paths by simply specifying a few parameters in drop-down menus. Paths can be visualized using four differents methods:</br>   
                1. <b><i>Sankey</i></b>: The Sankey diagram is a flow diagram used to visualize paths, transitions, or sequences of events in data. In the context of path analysis, a Sankey diagram helps represent how users, customers, or entities flow through different stages or events. The Sankey diagram merges user paths.</br>
                2. <b><i>Tree</i></b>: A Tree Diagram in path analysis is a hierarchical visualization of sequential events or decisions taken by users, customers, or entities. It is useful for understanding how different paths diverge and lead to specific outcomes. The Tree Diagram visualizes complete users’ paths and allows targeting underlying customers following a specific path to create a segment, which can then be stored in a table and leveraged for further analysis, personalization, or activation within a selected database and schema.</br>
                3. <b><i>Force Layout Graph</i></b>: A Force Layout Graph is a dynamic visualization used in path analysis to represent connections between events. It demonstrates the relationship between the different events that are part of the customer journey. This visualization provides a good understanding of which events are most closely related and which events most frequently lead to other specific events.</br>
                4. <b><i>Sunburst</i></b>: Compact yet informative, the Sunburst Diagram is a hierarchical, radial visualization used to display sequential paths or event flows. It is particularly useful in path analysis to understand nested relationships, user journeys or funnels. The Sunburst Diagram visualizes complete user paths and distribution percentages of each sequence.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
    <style>
        table {
            font-size: 13px !important;
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f4f4f4;
            font-weight: bold;
        }
    </style>

    <table>
        <tr>
            <th>Method</th>
            <th>Best For</th>
            <th>Limitations</th>
        </tr>
        <tr>
            <td><b>Sunburst Diagram</b></td>
            <td>Visualizing hierarchical paths</td>
            <td>Can be complex for very deep paths</td>
        </tr>
        <tr>
            <td><b>Sankey Diagram</b></td>
            <td>Tracking linear event transitions</td>
            <td>Hard to show nested relationships</td>
        </tr>
        <tr>
            <td><b>Tree Diagram</b></td>
            <td>Step-by-step hierarchical journeys</td>
            <td>Not as visually compact as Sunburst</td>
        </tr>
        <tr>
            <td><b>Force Graph</b></td>
            <td>Complex, interconnected paths</td>
            <td>Doesn’t clearly show sequences</td>
        </tr>
    </table>
""", unsafe_allow_html=True)
            
            st.markdown("""
            <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
                <b>Snowpath</b><sup>™</sup> similarly helps uncover key behavioral differences between path-based segments. The <span style="color:#29B5E8;">Compare</span> tab of the <span style="color:#29B5E8;">Path Analysis</span> page enables path-based behavioral analysis by comparing event sequences across two populations.</br> It offers two modes for comparison:</br>
                1. <b><i>Complement Mode</i></b>: This mode focuses on ensemble-based analysis by comparing a poluation defined by a reference set of paths against everything outside of it.</br>
                2. <b><i>Union Mode</i></b>: This mode performs a comparative analysis between two defined sets of paths (i.e., user groups/populations), accounting for any potential overlap.
            </h5>
            """, unsafe_allow_html=True)

            st.markdown("""
            <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
                The comparison output is a slope chart that compares the relative importance of events between two user groups (e.g., Churn vs Non-Churn), ranked by their TF-IDF scores derived from user journey paths. An upward slope indicates higher importance in the Non-Churn group, while a downward slope points to higher relevance in the Churn group. This helps quickly spot which actions differentiate the behaviors of the two populations.  
             </h5>
            """, unsafe_allow_html=True)

            st.markdown("""
            <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
                <b>Snowpath</b><sup>™</sup> also harnesses the interpretive and generative capabilities of LLMs through Snowflake AISQL to explain user journeys. This AI-powered explainability feature enables users to ask natural language questions about identified paths in order to summarize behaviors, uncover insights, and suggest potential actions or mitigations."  
             </h5>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
                Path analysis input data can be a table or view with events data. Minimum structure must include a unique identifier (text or numeric), an event and a timestamp. Additional input fields can be used to filter data.
             </h5>
            """, unsafe_allow_html=True)
            
        with st.expander("**Attribution Analysis**"):
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            (Marketing) Attribution modeling techniques aim to determine the contribution of each marketing touchpoint or channel in influencing customer behavior and driving conversions. These models provide valuable insights into the effectiveness of marketing efforts, helping businesses make informed decisions regarding resource allocation and optimization.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
           The concept of attribution extends far beyond marketing conversions. Attribution modeling can be applied to any event of interest, such as a customer signing up for a service, abandoning a cart, clicking on a specific feature, or even triggering an operational process. By analyzing the sequence of events leading up to a defined outcome, attribution techniques allow us to quantify the impact of each preceding touchpoint, whether it’s a website visit, an email click, or an app interaction.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            This broader application enables organizations across industries — such as e-commerce, IoT, manufacturing and any industry with complex operational workflows — to measure the contribution of key factors influencing desired or undesired outcomes and therefore optimize workflows, predict outcomes, preempt unwanted outcomes and finally make data-driven decisions that improve both user experience and operational efficiency. 
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            Rule-based attribution modeling relies on predetermined rules or heuristics to assign credit to various touchpoints along the customer journey. <b>Snowpath</b><sup>™</sup> rule-based models include the <b>First Touch</b>, <b>Last Touch</b>, <b>Uniform (linear)</b>, <b>U-shaped</b> and <b>Exponential (time decay)</b> models. The First Touch model attributes all credit to the first touchpoint a customer interacts with, while the Last Touch model assigns all credit to the final touchpoint before conversion. The Uniform model evenly distributes credit across all touchpoints in the customer journey. The Exponential model assigns more credit to touchpoints closer to the conversion event.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
             Algorithmic attribution modeling, in contrast, leverages advanced statistical and machine learning techniques to evaluate the contribution of each touchpoint. These models consider factors such as the sequence, timing, and interaction patterns of touchpoints. The <b>Snowpath</b><sup>™</sup> algorithmic model is built on the <b>Markov model</b>, a data-driven approach that applies the principles of Markov Chains to analyze marketing effectiveness. Markov Chains are mathematical frameworks that describe systems transitioning from one state to another in a sequential manner. This approach utilizes a transition matrix, derived from analyzing customer journeys from initial interactions to conversions or non-conversions, to capture the sequential nature of interactions and assess how each touchpoint influences the ultimate decision.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
             All approaches have their strengths and limitations. Rule-based models are relatively straight forward to implement and interpret, but they may oversimplify the complexity of customer journeys. Algorithmic-based models offer more data driven insights but may require advanced analytics expertise and extensive data sets to achieve accurate results.
             It's important for businesses to select the most suitable attribution modelling approach based on their specific goals, available data, and resources. Implementing an effective marketing attribution model can significantly enhance decision-making and optimize marketing strategies.
        </h5>
        """, unsafe_allow_html=True)
            
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
        The attribution analysis produces, in a detailed table, normalized scores to compare how different models value each event/touchpoint and calculates the “Time to Conversion” to understand delay or urgency post-event. 
        <br><br>
        <b>Count</b>: Total number of times this event occurred across all journeys.<br>
        <b>First Click</b>: Attribution score assigned if the event is the first in the journey.<br>
        <b>Last Click</b>: Attribution score given if the event was the last touchpoint before conversion.<br>
        <b>Uniform</b>: Attribution evenly distributed across all events in a journey.<br>
        <b>U-shape</b>: Gives more credit to the first (40) and last (40) touchpoints and shares the rest of the credit across the touchpoints in the middle.<br>
        <b>Exponential Decay</b>: Attribution giving more weight (0.5 decay) to recent events closer to conversion.<br>
        <b>Markov Conversion</b>: Probabilistic score based on the likelihood of conversion if the event is removed (based on Markov chain model).<br>
        <b>Time to Conversion (Sec/Min/Hour/Day)</b> Average time from when this event occurs to when the conversion happens, displayed in various time units for flexibility.
    </h5>
    """, unsafe_allow_html=True)
                
            st.markdown("""
    <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;"> 
        Another way to interpret the  attribution scores is to visualize them using charts. <b>Snowpath</b><sup>™</sup> 
        provides two additional ways to explore attribution analysis effectively:
        <br><br>
        1. <i><b>Bar Chart</b></i>: This Attribution Summary bar chart visually compares the attribution scores assigned to each event across different attribution models.
        <br>
        2. <i><b>Bubble Charts</b></i>: This section visualizes how each event/touchpoint performs across different attribution models, giving you a multidimensional view of their contribution. Each chart corresponds to one attribution model.
        The X-axis shows the Attribution Score (events on the right are more influential). The Y-axis shows the Time to Conversion (or how long users take to convert after this event : lower bubbles mean faster conversions - higher bubbles indicate events that take longer to result in a conversion.). 
        The Bubble size indicates the number of occurrences or how frequently this event occurred in journeys (larger = more common and could also means more costly)
    </h5>
    """, unsafe_allow_html=True)

            st.markdown("""
            <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
                <b>Snowpath</b><sup>™</sup> also leverages the interpretive and generative capabilities of LLMs through Snowflake AISQL to explain attribution results. This AI-powered explainability feature enables users to ask natural language questions about model outputs to understand how credit was assigned, identify key influencing factors, and suggest potential optimizations or next best actions."  
             </h5>
            """, unsafe_allow_html=True)
            
        with st.expander("**Association Analysis**"):
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            Association Analysis (also called association rules) is used for identifying patterns within groups of observations or transactions. It measures the relationships between items in these groups, such as how often certain items appear together or the likelihood that specific items will be present when other items are included in the group. Said differently, they are like if-then statements that show how likely it is that certain items are related in a large data set.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            The items, groups, and relationships depend on the nature of the data. Items can be products purchased and groups can be the market baskets in which they were purchased (market basket analysis). Similarly items can be events (interactions) and groups can be the customers who triggered them. 
        </h5>
        """, unsafe_allow_html=True)
            
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
             Association analysis is then useful in determining which products and services to sell with other products and services. This is why it is often used in applications like recommender engines for online retail sites. Association analysis can also be used to identify market trends, detect fraudulent activity, and understand customer behavior. More specifically in the context of events, it can help identify patterns where certain actions or behaviors are likely to occur together or sequentially, providing insights into relationships between user interactions.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
        The association analysis produces association rules (if-then statements) and measures of frequency, relationship, and statistical significance associated with these rules, in a detailed table. Below are the key metrics used in the analysis:
        <br><br>
        <b>Antecedent</b>: Represents the 'if' part of the rule, which is an item in the dataset.<br>
        <b>Consequent</b>: Represents the 'then' part of the rule, which is another item in the dataset.<br>
        <b>Cntb</b>: Represents the total co-occurrences of both antecedent and consequent items.<br>
        <b>Cnt1</b>: Represents the total occurrences of the antecedent item.<br>
        <b>Cnt2</b>: Represents the total occurrences of the consequent item.<br>
        <b>Score</b>: A calculated value that indicates the strength of association between two items (antecedent and consequent). It helps measure how often they co-occur compared to their individual occurrences.<br>
        <b>Confidence</b> (expressed as a percentage): Represents the probability that the consequent occurs given that the antecedent has occurred. A high confidence indicates that when the antecedent is present, the consequent is very likely to also be present.<br>
        <b>Support</b> (expressed as a percentage): epresents the proportion of transactions that contain both the antecedent and consequent items.<br>
        <b>Conviction</b> (expressed as a ratio): Measures how strongly the absence of the antecedent implies the absence of the consequent.<br>
        <b>Lift</b> (expressed as a ratio): Measures how much more likely the consequent is to occur when the antecedent is present compared to if they were independent.<br>
        <b>Z_Score</b> (expressed in standard deviations): Measures how statistically significant the observed co-occurrence is compared to what would be expected by chance. A higher absolute value of the Z_Score indicates a stronger and more statistically significant association between the items in the rule.
    </h5>
""", unsafe_allow_html=True)
            
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;"> 
            Thresholds can be applied to all metrics to filter the rules according to your criteria. 
            Another way to interpret the association rules metrics is to visualize them using charts. <b>Snowpath</b><sup>™</sup> 
            provides two additional ways to explore association patterns effectively:
            <br><br>
            1. <i><b>Heatmap</b></i>: The heatmap provides a structured grid-based visualization where rows and columns represent different items, 
            and color intensity indicates the strength of association based on the selected metric. This helps identify the most significant item pairs and 
            detect clusters of frequently co-occurring items at a glance.
            <br>
            2. <i><b>Force Layout Graph</b></i>: In the Force Layout Graph, each node represents an item (or event), and its size is proportional to the number of times it appears in the dataset. The edges between nodes represent associations between items, with thickness based on the total co-occurrences of both items. The color gradient of the edges is dynamically adjusted based on the selected metric, making it possible to visually distinguish the strength or significance of associations. This visualization allows for an interactive exploration of event relationships, highlighting the most frequent or impactful connections. Unlike proximity-based representations, the layout is influenced by a force-directed simulation, where repulsion and edge length settings help prevent excessive overlap, ensuring a clearer view of the association structure.
            <br><br>
            Together, these visualizations complement each other: the <b>Heatmap</b> offers a structured matrix-like view of associations, 
            while the <b>Force Layout Graph</b> provides an intuitive, interactive way to explore relationships dynamically.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
            <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
                <b>Snowpath</b><sup>™</sup> also leverages the interpretive and generative capabilities of LLMs through Snowflake AISQL to explain association rule results. This AI-powered explainability feature allows users to ask natural language questions about discovered patterns to better understand item relationships, extract actionable insights, and identify opportunities for cross-sell, upsell, or optimization."  
             </h5>
            """, unsafe_allow_html=True)
            
            st.markdown("""
    <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
        For more interactive and customizable network visualization and analysis, results can be exported into Gefx format and then imported into 
        <a href="https://gephi.org/gephi-lite/" target="_blank" style="color: #29B5E8; text-decoration: none; font-weight: bold;">
            Gephi Lite
        </a>. 
        Gephi Lite is a free and open-source web application to visualize and explore networks and graphs.
    </h5>
""", unsafe_allow_html=True)
            
        
        with st.expander("**Prediction Modeling**"):
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
         <b>Snowpath</b><sup>™</sup> Prediction Modeling creates and trains a machine learning model in Snowflake to predict outcomes based on user behavior paths and their underlying event sequences. It uses a binary classification ML function in Snowflake described here :<a>https://docs.snowflake.com/en/user-guide/ml-functions/classification</a>
        </h5>
        """, unsafe_allow_html=True)
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
          Classification involves creating a classification model object, passing in a reference to the training data. The model is fitted to the provided training data.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
          <b>Snowpath</b><sup>™</sup> classification model is stored in Snowflake with a given name upon creation. This allows  to conveniently explore predictions and ensures possible logging of the model within Snowflake Model Registry for later reusability. 
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
         For each model created, evaluation metrics, a confusion matrix, and feature importance insights are provided. 
        </h5>
        """, unsafe_allow_html=True)
#--------------------------------------
#APPLICATION MENU
#--------------------------------------

#def add_logo():
   # st.markdown(
    #    """
     #   <style>
      #  [data-testid="stSidebarNav"] {
       #     background-image: url("https://i.postimg.cc/XJnvCT6S/snowpathlogo2.png");
        #    background-repeat: no-repeat;
         #   background-size: contain;
          #  background-position: center;
           # padding-top: 350px; /* Adjust this */
            #margin: 10px;
        #}
        #</style>
        #""",
        #unsafe_allow_html=True,
   # )

#add_logo()

st.logo("snowpathlogo2.png",  size="large", link=None, icon_image="snowpathlogo2.png")

pg = st.navigation([
    #st.Page(home, title="Home", icon="*️⃣"),
    st.Page(home, title="Home", icon=":material/home:"),
    #st.Page("1_PathAnalysis.py", title="Path Analysis", icon="🔀"),
    st.Page("1_PathAnalysis.py", title="Path Analysis", icon=":material/conversion_path:"),
    #st.Page("2_AttributionAnalysis.py", title="Attribution Analysis", icon="📶"),
    st.Page("2_AttributionAnalysis.py", title="Attribution Analysis", icon=":material/bar_chart:"),
    #st.Page("3_AssociationAnalysis.py", title="Association Analysis", icon="🧑‍🧑‍🧒‍🧒"),
    st.Page("3_AssociationAnalysis.py", title="Association Analysis", icon=":material/graph_3:"),
    #st.Page("4_PredictiveModeling.py", title="Predictive Modeling", icon="⏭️"),
    st.Page("4_PredictiveModeling.py", title="Predictive Modeling", icon=":material/model_training:"),
    #st.Page(help, title="User Guide", icon="ℹ️")
    st.Page(help, title="User Guide", icon=":material/info:")
],expanded=False)
pg.run()

