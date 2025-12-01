# Import python packages
import json
from operator import iconcat
import streamlit as st

# Configure Streamlit theme - GitHub Quickstart Blue
st.set_page_config(
    page_title="Sequent",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)
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
                
                /* Light mode styles (default) */
                .custom-container,
                .custom-container-1 {
                    padding: 10px 10px 10px 10px;
                    border-radius: 10px;
                    background-color: #f7f7f7 !important; 
                    border: none;
                    margin-bottom: 20px;
                    transition: all 0.3s ease;
                }
                
                .custom-container h5,
                .custom-container-1 h5 {
                    font-size: 18px;
                    font-weight: normal;
                    color: #0f0f0f;
                    margin-top: 0px;
                    margin-bottom: -15px;
                }
                
                .footer {
                    position: fixed;
                    bottom: 0;
                    left: 0;
                    width: 100%;
                    background-color: #f7f7f7;
                    color: #666;
                    padding: 10px;
                    font-size: 12px;
                    text-align: center;
                    z-index: 999;
                    border-top: 1px solid #ccc;
                    transition: all 0.3s ease;
                }
                
                /* Streamlit native dark mode detection */
                @media (prefers-color-scheme: dark) {
                    .custom-container-1 {
                        background-color: transparent !important;
                        border: 1px solid #4a4a4a !important;
                    }
                    
                    .custom-container-1 h5 {
                        color: #b0b0b0 !important;
                    }
                    
                    /* Welcome message text color in dark mode */
                    [data-testid="stMarkdownContainer"] p {
                        color: #b0b0b0 !important;
                    }
                    
                    .footer {
                        background-color: #0e1117 !important;
                        color: #cccccc !important;
                        border-top: 1px solid #262730 !important;
                    }
                    
                    /* Ensure expander content text is readable in dark mode */
                    .stExpander [data-testid="stMarkdownContainer"] h5,
                    .stExpander [data-testid="stMarkdownContainer"] p {
                        color: #b0b0b0 !important;
                    }
                    
                    /* Fix table styling in dark mode */
                    .stExpander table {
                        border-collapse: collapse !important;
                        width: 100% !important;
                        background-color: transparent !important;
                    }
                    
                    .stExpander table th {
                        background-color: #4a4a4a !important;
                        color: #ffffff !important;
                        font-weight: bold !important;
                        padding: 12px 8px !important;
                        text-align: left !important;
                        border: 1px solid #4a4a4a !important;
                    }
                    
                    .stExpander table td {
                        background-color: transparent !important;
                        color: #ffffff !important;
                        padding: 10px 8px !important;
                        border: 1px solid #555 !important;
                    }
                    
                    .stExpander table td b {
                        color: #ffffff !important;
                    }
                }
                
        </style>
        """, unsafe_allow_html=True)
   
#--------------------------------------
#HOME PAGE
#--------------------------------------
def home():
        # Custom button styling and card shadows for home page only
        st.markdown("""
        <style>
        /* Light mode button styling */
        button[kind="secondary"] {
            background-color: #f7f7f7 !important;
            color: #262730 !important;
            border: 1px solid #d0d0d0 !important;
        }
        
        button[kind="secondary"]:hover {
            background-color: #ffffff !important;
            color: #262730 !important;
            border: 1px solid #b0b0b0 !important;
        }
        
        /* Dark mode - maintain button visibility */
        @media (prefers-color-scheme: dark) {
            button[kind="secondary"] {
                background-color: #1e232a !important;
                color: #ffffff !important;
                border: 1px solid #4a4a4a !important;
            }
            
            button[kind="secondary"]:hover {
                background-color: #2d333a !important;
                color: #ffffff !important;
                border: 1px solid #666666 !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.write("")
        # Full-width bordered container with centered logo
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image("Sequent.png", use_container_width=True)
        
        # Centered welcome message
        st.markdown("""
        <p style='text-align: center; font-size: 18px; color: #666; margin-top: 30px; margin-bottom: 10px; font-weight: bold;'>
        WELCOME TO SEQUENT<sup>™</sup>, YOUR ONE-STOP SHOP SNOWFLAKE NATIVE APPLICATION FOR BEHAVIORAL INTELLIGENCE.
        </p>
        <p style='text-align: center; font-size: 18px; color: #666; margin-top: 0px; margin-bottom: 40px; font-weight: bold;'>
        DECODE YOUR CUSTOMER JOURNEYS AND UNLOCK INSIGHTS, RIGHT WHERE YOUR DATA LIVES.
        </p>
        """, unsafe_allow_html=True)
        

        
        # Module cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center; margin-top: 20px;'>Path Analysis</h3>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Uncover and compare behavioral journeys.</p>", unsafe_allow_html=True)
                if st.button("Analyze Paths", key="path_btn", use_container_width=True):
                    st.switch_page("1_PathAnalysis.py")
        
        with col2:
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center; margin-top: 20px;'>Pattern Mining</h3>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Discover frequent sequences and timing patterns.</p>", unsafe_allow_html=True)
                if st.button("Explore Patterns", key="pattern_btn", use_container_width=True):
                    st.switch_page("4_PatternMining.py")
        
        with col3:
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center; margin-top: 20px;'>Attribution Analysis</h3>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Measure touchpoint contribution to outcomes.</p>", unsafe_allow_html=True)
                if st.button("Analyze Attribution", key="attr_btn", use_container_width=True):
                    st.switch_page("2_AttributionAnalysis.py")

        
        st.write("")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center; margin-top: 20px;'>Behavioral Segmentation</h3>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Group customers by journey patterns.</p>", unsafe_allow_html=True)
                if st.button("Segment Customers", key="segment_btn", use_container_width=True):
                    st.switch_page("6_BehavioralSegmentation.py")
        
        with col5:
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center; margin-top: 20px;'>Association Analysis</h3>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Uncover relationships between events.</p>", unsafe_allow_html=True)
                if st.button("Find Associations", key="assoc_btn", use_container_width=True):
                    st.switch_page("3_AssociationAnalysis.py")
        
        with col6:
            with st.container(border=True):
                st.markdown("<h3 style='text-align: center; margin-top: 20px;'>Predictive Modeling</h3>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center;'>Anticipate outcomes based on sequence history.</p>", unsafe_allow_html=True)
                if st.button("Predict Outcomes", key="predict_btn", use_container_width=True):
                    st.switch_page("5_PredictiveModeling.py")
        
        st.write("")
        st.write("")
        
        # About section - original content
        with st.expander("**ABOUT**"):
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
             <b>Sequent</b><sup>™</sup> native application allows users to easily and visually perform and deep dive into <b>Path Analysis</b>, <b>Attribution Analysis</b>, <b>Association Analysis</b>, <b>Pattern Mining</b>, and <b>Behavioral Segmentation</b> by simply specifying a few parameters in drop-down menus. Leveraging advanced techniques, <b>Sequent</b><sup>™</sup> intuitively and visually helps identify touchpoints influencing customer (or machine) behaviours, targets them to create segments, performs cross-population behavioural comparisons, computes rule-based and ML-driven attribution models to understand the contribution of each event preceding a specific outcome, conducts association analysis to uncover hidden patterns and relationships between events, discovers frequent sequential patterns and behavioral signatures through advanced pattern mining, and enables sophisticated behavioral segmentation to group customers based on their journey patterns and characteristics. <b>Sequent</b><sup>™</sup> also harnesses the interpretive and generative power of LLMs thanks to Snowflake AISQL to explain journeys, attribution models, association rules, pattern insights and derive insights (summarize and analyze results, describe behaviors and even suggest actions !)
        </h5>
        """, unsafe_allow_html=True)
            
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
             Visualizing and identifying paths can itself be actionable and often uncovers an area of interest for additional analysis. First, the picture revealed by path analysis can be further enriched with attribution analysis, association analysis, pattern mining, and behavioral segmentation. Attribution helps quantify the contribution of individual touchpoints to a defined outcome, association analysis uncovers relationships between events that frequently occur together, pattern mining discovers frequent sequential behaviors and hidden temporal dependencies, and behavioral segmentation groups customers into meaningful clusters based on their journey characteristics and patterns. Together, these techniques provide a comprehensive understanding of event sequences, enabling data-driven decision-making and uncovering new opportunities for optimization. Second, path insights can be used directly to predict outcomes (<b>Predictive Modeling</b>) or to derive behavioral features (such as the frequency of specific patterns and sequence signatures). These features can then be integrated into existing predictive models, enhancing their accuracy and enabling deeper customer understanding through advanced segmentation strategies.
        </h5>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="footer">
            Copyright &copy; 2025 Yannis Marigo. All rights reserved.
        </div>
        """, unsafe_allow_html=True)

#--------------------------------------
#HELP PAGE
#--------------------------------------
def help():
        # Page Title - using centralized styling
        st.markdown("""
        <div class="custom-container-1">
            <h5 style="font-size: 18px; font-weight: normal; margin-top: 0px; margin-bottom: -15px;">
                USER GUIDE
            </h5>
         </div>
            """, unsafe_allow_html=True)

        with st.expander("Path Analysis", icon=":material/conversion_path:"):
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
             <b>Sequent</b><sup>™</sup> helps identifying paths and patterns in data in a valuable way to gain insight into the occurrences leading to or from any event of interest or between two events of interest. In the Analyze tab of the Path Analysis page, users can visually dive deep into paths by simply specifying a few parameters in drop-down menus. Paths can be visualized using four differents methods:</br>   
                1. <b><i>Sankey</i></b>: The Sankey diagram is a flow diagram used to visualize paths, transitions, or sequences of events in data. In the context of path analysis, a Sankey diagram helps represent how users, customers, or entities flow through different stages or events. The Sankey diagram merges user paths. Single-event paths are not displayed in a Sankey diagram, use the Sunburst Diagram instead.</br>
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
                <b>Sequent</b><sup>™</sup> similarly helps uncover key behavioral differences between path-based segments. The Compare tab of the Path Analysis page enables path-based behavioral analysis by comparing event sequences across two populations.</br> It offers two modes for comparison:</br>
                1. <b><i>Complement Mode</i></b>: This mode focuses on ensemble-based analysis by comparing a poluation defined by a reference set of paths against everything outside of it.</br>
                2. <b><i>Union Mode</i></b>: This mode performs a comparative analysis between two defined sets of paths (i.e., user groups/populations), accounting for any potential overlap.
            </h5>
            """, unsafe_allow_html=True)

            st.markdown("""
            <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
                The comparison output provides two complementary visualizations:</br>
                1. <b><i>Event Ranking Comparison (Slope Chart)</i></b>: Compares the relative importance of events between two user groups (e.g., Reference vs Compared), ranked by their TF-IDF scores derived from user journey paths. An upward slope indicates higher importance in the Reference group, while a downward slope points to higher relevance in the Compared group. This helps quickly spot which actions differentiate the behaviors of the two populations.</br>
                2. <b><i>Path Comparison (Side-by-Side Sunburst Charts)</i></b>: Displays hierarchical sunburst visualizations side-by-side to compare the actual path structures between the two populations. Each sunburst shows the sequential flow and distribution of customer journeys, with consistent color coding across both charts to enable easy visual comparison of path patterns, frequencies, and user distributions between the reference and compared groups.
             </h5>
            """, unsafe_allow_html=True)

            st.markdown("""
            <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
                <b>Sequent</b><sup>™</sup> also harnesses the interpretive and generative capabilities of LLMs through Snowflake AISQL to explain user journeys. This AI-powered explainability feature enables users to ask natural language questions about identified paths in order to summarize behaviors, uncover insights, and suggest potential actions or mitigations."  
             </h5>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
                Path analysis input data can be a table or view with events data. Minimum structure must include a unique identifier (text or numeric), an event and a timestamp. Additional input fields can be used to filter data.
             </h5>
            """, unsafe_allow_html=True)
            
        with st.expander("Attribution Analysis", icon=":material/bar_chart:"):
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
            Rule-based attribution modeling relies on predetermined rules or heuristics to assign credit to various touchpoints along the customer journey. <b>Sequent</b><sup>™</sup> rule-based models include the <b>First Touch</b>, <b>Last Touch</b>, <b>Uniform (linear)</b>, <b>U-shaped</b> and <b>Exponential (time decay)</b> models. The First Touch model attributes all credit to the first touchpoint a customer interacts with, while the Last Touch model assigns all credit to the final touchpoint before conversion. The Uniform model evenly distributes credit across all touchpoints in the customer journey. The Exponential model assigns more credit to touchpoints closer to the conversion event.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
             Algorithmic attribution modeling, in contrast, leverages advanced statistical and machine learning techniques to evaluate the contribution of each touchpoint. These models consider factors such as the sequence, timing, and interaction patterns of touchpoints. <b>Sequent</b><sup>™</sup> algorithmic models include the <b>Markov model</b> and the <b>Shapley Value model</b>. The Markov model is a data-driven approach that applies the principles of Markov Chains to analyze marketing effectiveness. Markov Chains are mathematical frameworks that describe systems transitioning from one state to another in a sequential manner. This approach utilizes a transition matrix, derived from analyzing customer journeys from initial interactions to conversions or non-conversions, to capture the sequential nature of interactions and assess how each touchpoint influences the ultimate decision. The Shapley Value model, derived from cooperative game theory, provides a fair allocation of credit by considering the marginal contribution of each touchpoint across all possible combinations of events in the customer journey.
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
        <b>Shapley Value</b>: Fair attribution score derived from cooperative game theory, considering the marginal contribution of each event across all possible combinations in the customer journey.<br>
        <b>Time to Conversion (Sec/Min/Hour/Day)</b> Average time from when this event occurs to when the conversion happens, displayed in various time units for flexibility.
    </h5>
    """, unsafe_allow_html=True)
                
            st.markdown("""
    <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;"> 
        Another way to interpret the  attribution scores is to visualize them using charts. <b>Sequent</b><sup>™</sup> 
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
                <b>Sequent</b><sup>™</sup> also leverages the interpretive and generative capabilities of LLMs through Snowflake AISQL to explain attribution results. This AI-powered explainability feature enables users to ask natural language questions about model outputs to understand how credit was assigned, identify key influencing factors, and suggest potential optimizations or next best actions."  
             </h5>
            """, unsafe_allow_html=True)
            
        with st.expander("Association Analysis", icon=":material/graph_3:"):
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
            Another way to interpret the association rules metrics is to visualize them using charts. <b>Sequent</b><sup>™</sup> 
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
                <b>Sequent</b><sup>™</sup> also leverages the interpretive and generative capabilities of LLMs through Snowflake AISQL to explain association rule results. This AI-powered explainability feature allows users to ask natural language questions about discovered patterns to better understand item relationships, extract actionable insights, and identify opportunities for cross-sell, upsell, or optimization."  
             </h5>
            """, unsafe_allow_html=True)
            
            st.markdown("""
    <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
        For more interactive and customizable network visualization and analysis, results can be exported into Gefx format and then imported into 
        <a href="https://gephi.org/gephi-lite/" target="_blank" text-decoration: none; font-weight: bold;">
            Gephi Lite
        </a>. 
        Gephi Lite is a free and open-source web application to visualize and explore networks and graphs.
    </h5>
""", unsafe_allow_html=True)
            
        with st.expander("Pattern Mining", icon=":material/auto_awesome:"):
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            Pattern Mining is an advanced analytical technique that discovers frequent sequential patterns and behavioral sequences within event data. It identifies hidden temporal dependencies, common subsequences, and recurring behavioral patterns across user journeys or process flows. Unlike path analysis which focuses on specific start-to-end journeys, pattern mining explores all possible sequential combinations to uncover the most significant behavioral signatures in your data.
        </h5>
        """, unsafe_allow_html=True)
            
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            <b>Sequent</b><sup>™</sup> Pattern Mining offers four distinct pattern definition modes to capture different types of behavioral insights:
            <br><br>
            1. <b><i>All Patterns</i></b>: Discovers all possible sequential patterns of specified lengths, providing comprehensive coverage of behavioral sequences without constraints.
            <br>
            2. <b><i>Contains Patterns</i></b>: Identifies patterns that must include specific events, regardless of their position in the sequence, enabling focused analysis on key touchpoints.
            <br>
            3. <b><i>Starts With Patterns</i></b>: Finds sequences that begin with particular events, ideal for understanding common continuation paths after specific triggers or entry points.
            <br>
            4. <b><i>Ends With Patterns</i></b>: Discovers patterns concluding with specific events, perfect for analyzing common precursor sequences leading to important outcomes or conversions.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            The pattern mining engine uses sophisticated SQL MATCH_RECOGNIZE functionality with configurable sequence sizes (minimum/maximum ranges or exact lengths) and optional time-gap constraints to ensure patterns represent realistic behavioral windows. Advanced features include display column mapping for cleaner visualizations and comprehensive filtering capabilities for focused analysis.
        </h5>
        """, unsafe_allow_html=True)
            
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            Pattern mining results include detailed statistical analysis with frequency counts, unique user reach, percentage distributions, and sequence length analytics. Interactive visualizations provide multiple perspectives including frequency rankings, pattern length distributions, and reach-versus-frequency scatter plots to identify the most impactful behavioral patterns.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            <b>AI-Powered Insights:</b> <b>Sequent</b><sup>™</sup> leverages Snowflake Cortex LLMs with dynamic model selection (including Mixtral, Mistral, Llama, and Gemma models) to provide intelligent explanations of discovered patterns. The AI analysis interprets statistical findings within business context, suggesting optimization opportunities, identifying behavioral anomalies, and providing actionable recommendations for customer journey improvement.
        </h5>
        """, unsafe_allow_html=True)
        
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            <b>Use Cases and Applications:</b> These insights enable powerful applications including customer journey optimization, process flow analysis, behavioral segmentation, anomaly detection, feature engineering for predictive modeling, marketing campaign sequence design, and operational workflow improvement. The combination of statistical pattern discovery and AI interpretation makes complex behavioral data accessible and actionable for business users.
        </h5>
        """, unsafe_allow_html=True)
        
        with st.expander("Prediction Modeling", icon=":material/model_training:"):
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
         <b>Sequent</b><sup>™</sup> Predictive Modeling transforms customer journey data into powerful predictive insights by creating and training machine learning models directly in Snowflake. It leverages <b>Snowpark ML</b> and <b>Scikit-learn</b> classification algorithms to predict binary outcomes based on user behavior paths and their underlying event sequences. Users can choose from multiple algorithms optimized for different scenarios and compute requirements.
        </h5>
        """, unsafe_allow_html=True)
            
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
          <b>Available Algorithms:</b></br>
          • <b>Naive Bayes (Bernoulli)</b>: Uses binary features representing event presence/absence. Fast training, ideal for large datasets with sparse features. Suitable for standard compute warehouses.</br>
          • <b>Naive Bayes (Multinomial)</b>: Preserves TF-IDF values to capture event importance and frequency. Better for understanding event weight in predictions.</br>
          • <b>Random Forest</b>: Ensemble learning with multiple decision trees. Provides feature importance rankings and handles non-linear relationships. More compute-intensive; consider Snowpark-optimized warehouses for large datasets.</br>
          Each algorithm is available in both <b>Snowpark ML</b> (native Snowflake execution) and <b>Scikit-learn</b> (Python-based) implementations.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
          <b>How to Use Predictive Modeling:</b></br>
          1. <b><i>Data Preparation</i></b>: Start with event data containing customer journeys, ensuring you have a clear binary target variable (e.g., churned/not churned, converted/not converted).</br>
          2. <b><i>Define Classes</i></b>: Specify the reference (primary) class and comparison (complementary) class for your binary classification task.</br>
          3. <b><i>Feature Engineering</i></b>: The system automatically transforms event sequences into TF-IDF features, capturing both event occurrence and relative importance across customer journeys.</br>
          4. <b><i>Algorithm Selection</i></b>: Choose your classification algorithm based on dataset size, compute resources, and interpretability needs. Configure Random Forest hyperparameters (n_estimators, max_depth) if selected.</br>
          5. <b><i>Model Training</i></b>: Models train on an 80/20 train/test split. Monitor progress through the Model Training tab.</br>
          6. <b><i>Evaluation & Results</i></b>: Review performance metrics (accuracy, precision, recall, F1-score), confusion matrix, classification report, and feature importance in the Results tab.</br>
          7. <b><i>Model Registry</i></b>: Optionally log your trained model to Snowflake Model Registry with version control, metadata, and comments in the Model Logging tab.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
          <b>Performance Metrics & Evaluation:</b></br>
          • <b>Test Accuracy</b>: Overall prediction accuracy on held-out test data (sampled for performance optimization).</br>
          • <b>Classification Report</b>: Detailed precision, recall, F1-score, and support for each class.</br>
          • <b>Confusion Matrix</b>: Visual heatmap showing true positives, false positives, true negatives, and false negatives.</br>
          • <b>Feature Importance</b>: (Random Forest only) Ranking of most influential events in predictions with statistical analysis.</br>
          • <b>Sampling Controls</b>: Configure sample percentage and maximum rows for metrics computation to balance accuracy and performance.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
          <b>Model Registry & Deployment:</b></br>
          Trained models can be logged to Snowflake's Model Registry with versioning (V1, V2, etc.), accuracy metrics, and custom comments. This enables model governance, reusability across teams, and streamlined deployment for real-time scoring on new customer journeys. Registered models can be retrieved programmatically for integration into production pipelines.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
          <b>Common Use Cases:</b> Churn prediction (identify customers likely to leave), conversion forecasting (predict purchase likelihood), engagement scoring (assess customer activity levels), risk assessment (detect potentially problematic behaviors), campaign targeting (identify high-value prospects), and retention modeling (predict customer lifetime value). The combination of Snowpark ML's scalability and Scikit-learn's flexibility ensures optimal performance for diverse use cases.
        </h5>
        """, unsafe_allow_html=True)
        
        with st.expander("Behavioral Segmentation", icon=":material/groups:"):
            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            Behavioral Segmentation identifies distinct groups of customers based on their interaction patterns and event sequences. By analyzing how customers behave across touchpoints, businesses can create targeted segments for personalized marketing, product development, and customer experience optimization.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            <b>Sequent</b><sup>™</sup> Behavioral Segmentation uses advanced machine learning techniques to transform customer event sequences into meaningful behavioral patterns. The process involves two main approaches: <b>Event2Vec clustering</b> for embedding-based analysis and <b>Latent Dirichlet Allocation (LDA)</b> for topic-based segmentation.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            <b>Event2Vec Clustering</b> transforms customer event sequences into high-dimensional vector representations using Word2Vec embeddings. This approach treats customer journeys as "sentences" and individual events as "words," learning semantic relationships between events based on their co-occurrence patterns. The resulting embeddings capture behavioral similarities and enable sophisticated clustering algorithms to identify customer segments.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            <b>Available Clustering Methods:</b></br>
            • <b>K-Means</b>: Partitions customers into k distinct clusters based on behavioral similarity. Best for well-separated, spherical segments.</br>
            • <b>DBSCAN</b>: Density-based clustering that identifies core behavioral patterns and outliers. Excellent for discovering irregular segment shapes and detecting anomalous behavior.</br>
            • <b>Gaussian Mixture</b>: Probabilistic clustering that models segments as overlapping distributions. Ideal when customers may belong to multiple segments with varying degrees of membership.</br>
            • <b>Hierarchical</b>: Creates a tree-like structure of nested segments, revealing behavioral hierarchies and sub-segments within larger groups.</br>
            • <b>Latent Dirichlet Allocation (LDA)</b>: Topic modeling approach that identifies behavioral "topics" or themes. Each customer is represented as a mixture of behavioral topics, allowing for nuanced segment interpretation.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            <b>Feature Engineering Options:</b></br>
            • <b>Vector Dimensions (50-300)</b>: Controls the complexity of event embeddings. Higher dimensions capture more nuanced relationships but require more data.</br>
            • <b>Context Window (2-10)</b>: Determines how many surrounding events are considered when learning relationships. Larger windows capture longer-term behavioral patterns.</br>
            • <b>Feature Weighting</b>: Multiple strategies for emphasizing event importance:</br>
            &nbsp;&nbsp;- <b>TF-IDF</b>: Emphasizes distinctive events that differentiate customer segments</br>
            &nbsp;&nbsp;- <b>Smart Filtering</b>: Automatically removes noise and focuses on meaningful behavioral signals</br>
            &nbsp;&nbsp;- <b>Raw Frequency</b>: Uses simple event counts</br>
            &nbsp;&nbsp;- <b>Log Frequency</b>: Reduces the impact of very frequent events</br>
            &nbsp;&nbsp;- <b>Binary Presence</b>: Focuses on event occurrence rather than frequency
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            <b>Principal Component Analysis (PCA)</b> reduces data complexity while preserving important behavioral patterns:</br>
            • <b>No PCA</b>: Uses all original event features, preserving complete information but potentially including noise</br>
            • <b>Auto (Recommended)</b>: Automatically selects optimal components based on the clustering method (90% variance for K-Means/Gaussian, 95% for DBSCAN/Hierarchical)</br>
            • <b>Manual</b>: Allows precise control over dimensionality reduction with 1-50 components
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            <b>Quality Metrics and Evaluation:</b></br>
            • <b>Silhouette Score</b>: Measures how well-separated and cohesive clusters are (-1 to 1, higher is better)</br>
            • <b>Calinski-Harabasz Index</b>: Evaluates cluster separation relative to within-cluster dispersion</br>
            • <b>Davies-Bouldin Index</b>: Measures average similarity between clusters (lower is better)</br>
            • <b>LDA-Specific Metrics</b>: Perplexity, log-likelihood, topic coherence, and assignment clarity for topic modeling evaluation
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            <b>Visualization and Insights:</b></br>
            • <b>Interactive Bar Charts</b>: Show cluster size distribution and customer counts</br>
            • <b>Multi-level Treemaps</b>: Hierarchical visualization of clusters and their top events, with customizable depth</br>
            • <b>AI-Powered Analysis</b>: Leverages Snowflake Cortex LLMs to provide business insights, segment interpretations, and actionable recommendations</br>
            • <b>Cluster Interpretation</b>: AI analysis of each segment's behavioral characteristics and business implications
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            <b>Segment Activation:</b> Results can be written back to Snowflake tables for activation in downstream systems. Users can select specific clusters, choose target databases and schemas, and create persistent segment tables for marketing automation, personalization engines, or further analysis.
        </h5>
        """, unsafe_allow_html=True)

            st.markdown("""
        <h5 style="font-size: 13px; font-weight: normal; color: black; margin-top: 0px; margin-bottom: -15px;">
            <b>Use Cases:</b> E-commerce customer segmentation, subscription behavior analysis, user engagement patterns, churn prediction segments, product recommendation groups, marketing campaign targeting, customer lifetime value segments, and operational workflow optimization.
        </h5>
        """, unsafe_allow_html=True)

st.logo("Sequent.png",  size="large", link=None, icon_image="Sequent.png")

pg = st.navigation([
    st.Page(home, title="Home", icon=":material/home:"),
    st.Page("1_PathAnalysis.py", title="Path Analysis", icon=":material/conversion_path:"),
    st.Page("4_PatternMining.py", title="Pattern Mining", icon=":material/auto_awesome:"),
    st.Page("2_AttributionAnalysis.py", title="Attribution Analysis", icon=":material/bar_chart:"),
    st.Page("3_AssociationAnalysis.py", title="Association Analysis", icon=":material/graph_3:"),
    st.Page("6_BehavioralSegmentation.py", title="Behavioral Segmentation", icon=":material/groups:"),
    st.Page("5_PredictiveModeling.py", title="Predictive Modeling", icon=":material/model_training:"),
    st.Page(help, title="User Guide", icon=":material/quick_reference:")
],expanded=False)
pg.run()


