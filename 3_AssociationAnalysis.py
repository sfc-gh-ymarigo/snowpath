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
from snowflake.snowpark import Session#
from collections import defaultdict
from collections import Counter
from matplotlib.colors import Normalize
from streamlit_echarts import st_echarts
import math
import ast
from streamlit_extras.app_logo import add_logo

# Call function to create new or get existing Snowpark session to connect to Snowflake
session = get_active_session()

#st.set_page_config(layout="wide")

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
#ASSOCIATION ANALYSIS
#--------------------------------------
def rgba_to_str(rgba):
    return f"rgba({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)}, {rgba[3]})"
        
def sigma_graph(df, metric):
    # Step 1: Extract event pairs and calculate weights
    event_pairs = []
    event_counts = Counter()

    for _, row in df.iterrows():
        antecedent = row["ANTECEDENT"]
        consequent = row["CONSEQUENT"]
        cnt1 = row["CNT1"]
        cntb = row["CNTB"]
        event_pairs.append((antecedent, consequent, cntb))
        event_counts[antecedent] += cnt1
        event_counts[consequent] += cnt1

    pair_counts = Counter({(pair[0], pair[1]): pair[2] for pair in event_pairs})
    total_events = sum(event_counts.values())
    total_pairs = sum(pair_counts.values())

    # Step 2: Create nodes with size based on event counts
    unique_events = list(event_counts.keys())
    max_count = max(event_counts.values())  # Normalize node size for visualization

    nodes = [
        {
            "name": event,
            "symbolSize": 10 + (event_counts[event] / max_count) * 50,  # Base size + scaled size
            "itemStyle": {
                "color": f"hsl({i * 360 / len(unique_events)}, 70%, 50%, 0.7)"  # Add transparency (0.7)
            },
            "label": {
                "show": True,
                "position": "right",
                "color": "#000000",  # Black font for simplicity
                "fontWeight": "normal",  # Bold font for node labels
            },
            "emphasis": {
                "itemStyle": {
                    "color": f"hsl({i * 360 / len(unique_events)}, 70%, 50%, 0.7)"  # Maintain same color on hover
                },
                "label": {
                    "color": "#000000",  # Keep the label color black on hover
                    "fontWeight": "n"
                }
            },
            "tooltip": {
                "formatter": f"{event}<br>Count: {event_counts[event]}<br>Percentage: {event_counts[event] / total_events:.2%}"
            },
        }
        for i, event in enumerate(unique_events)
    ]

    # Step 3: Create edges with logarithmic thickness based on count
    # **Normalize color scaling based on selected metric**
    min_color_value = df[metric].min()
    max_color_value = df[metric].max()
    norm = Normalize(vmin=min_color_value, vmax=max_color_value)  # Dynamically scale colors
    max_width=6
        
    edges = [
    {
        "source": src,
        "target": tgt,
        "lineStyle": {
            "width": min(np.log(cntb + 1), max_width),
            "color": rgba_to_str(plt.cm.Blues(norm(row[metric])))  # Correctly access the metric
        },
        "tooltip": {
            "formatter": f"{src} → {tgt}<br>Association Count: {cntb}<br>{metric}: {row[metric]:.4f}"
        }
    }
    for _, row in df.iterrows()  # Ensure row is a Pandas Series, not a tuple
    for src, tgt, cntb in [(row["ANTECEDENT"], row["CONSEQUENT"], row["CNTB"])]
    ]
    # Step 4: ECharts options for force-directed graph
    options = {
        "tooltip": {"trigger": "item"},
        "series": [
            {
                "type": "graph",
                "layout": "force",
                "symbolSize": 20,
                "roam": True,
                "label": {"show": True, "fontSize": 12},
                "edgeSymbol": ["circle", "arrow"],
                "edgeSymbolSize": [4, 10],
                "force": {
                    "repulsion": 500,  # Increase repulsion to reduce overlap
                    "edgeLength": [50, 200],  # Set minimum and maximum edge lengths
                },
                "data": nodes,
                "links": edges,
                "lineStyle": {
                    "curveness": 0.1,  # Slight curvature for readability
                    "opacity": 0.7,    # Slight transparency for overlapping edges
                },
                "emphasis": {
                    "focus": "adjacency",
                    "label": {
                        "color": "#ff0000",  # Highlight label color on hover
                        "fontWeight": "bold"
                    }
                },
            }
        ],
    }

    # Step 5: Render the ECharts graph in Streamlit
    st_echarts(options=options, height="800px")
    

    
st.sidebar.markdown("")
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
        ASSOCIATION ANALYSIS
    </h5>
</div>
""", unsafe_allow_html=True)
# Get the current credentials
session = get_active_session()
#Initialize variables
uid = None
evt = None
tbl = None
partitionby = None
startdt_input = None
enddt_input = None
sess = None
excl3 = "''"
cols=''
colsdf=pd.DataFrame()
with st.expander("Input parameters"):
        
        # DATA SOURCE 
        st.markdown("""
    <h2 style='font-size: 14px; margin-bottom: 0px;'>Data Source</h2>
    <hr style='margin-top: -8px;margin-bottom: 5px;'>
    """, unsafe_allow_html=True)
        
            # Get list of databases
        sqldb = "SHOW DATABASES"
        databases = session.sql(sqldb).collect()
        db0 = pd.DataFrame(databases)
            
        col1, col2, col3 = st.columns(3)
            
            # **Database Selection**
        with col1:
            database = st.selectbox('Select Database', key='assodb', index=None, 
                                        placeholder="Choose from list...", options=db0['name'].unique())
            
            # **Schema Selection (Only if a database is selected)**
        if database:
            sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
            schemas = session.sql(sqlschemas).collect()
            schema0 = pd.DataFrame(schemas)
            
            with col2:
                schema = st.selectbox('Select Schema', key='assosch', index=None, 
                                          placeholder="Choose from list...", options=schema0['name'].unique())
        else:
            schema = None  # Prevents SQL execution
            
            # **Table Selection (Only if a database & schema are selected)**
        if database and schema:
            sqltables = f"""
                    SELECT TABLE_NAME 
                    FROM {database}.INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = '{schema}' AND TABLE_TYPE IN ('BASE TABLE', 'VIEW')
                """
            tables = session.sql(sqltables).collect()
            table0 = pd.DataFrame(tables)
            
            with col3:
                tbl = st.selectbox('Select Event Table or View', key='assotbl', index=None, 
                                       placeholder="Choose from list...", options=table0['TABLE_NAME'].unique(),
                                       help="Select any table or view with events data. Minimum structure should include a unique identifier, an event, and a timestamp. Additional fields can be used to filter data.")
        else:
            tbl = None  # Prevents SQL execution
            
            # **Column Selection (Only if a database, schema, and table are selected)**
        if database and schema and tbl:
            cols = f"""
                    SELECT COLUMN_NAME
                    FROM {database}.INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{tbl}'
                    ORDER BY ORDINAL_POSITION;
                """

            colssql = session.sql(cols).collect()
            colsdf = pd.DataFrame(colssql)
        col3, col4, col5 = st.columns(3)
        with col3:
            uid = st.selectbox('Select identifier column', colsdf, index=None, placeholder="Choose from list...",help="The identifier column is the unique identifier for each record. This may be a user id, a customer id, a process id, a basket id, a sale id, etc")
        with col4: 
            item = st.selectbox('Select item column', colsdf, index=None, placeholder="Choose from list...", help="The item column : a product, an event, a channel, a service or even a person,...)")
        
if (uid != None and item != None):
    
    associationsql=f"""WITH
CNT1 AS (
    SELECT SUM(cnt1) AS cnt1, {item}
    FROM (
        SELECT COUNT({item}) AS cnt1, {item}, {uid}
        FROM {database}.{schema}.{tbl}
        GROUP BY {uid}, {item}
    )
    GROUP BY {item}
),
CNT2 AS (
    SELECT SUM(cnt2) AS cnt2, {item}
    FROM (
        SELECT COUNT({item}) AS cnt2, {item}, {uid}
        FROM {database}.{schema}.{tbl}
        GROUP BY {uid}, {item}
    )
    GROUP BY {item}
),
CNTB AS (
    SELECT item1, item2, SUM(cnt) AS cntb
    FROM (
        SELECT COUNT(DISTINCT A.{item}, B.{item}) AS cnt, A.{item} AS item1, B.{item} AS item2, A.{uid}
        FROM {database}.{schema}.{tbl} A
        JOIN {database}.{schema}.{tbl} B
            ON A.{uid} = B.{uid}
            AND A.{item} <> B.{item}
        GROUP BY A.{uid}, A.{item}, B.{item}
    )
    GROUP BY item1, item2
),
NUMPART AS (
    SELECT DISTINCT COUNT(DISTINCT {uid}) OVER (PARTITION BY 1) AS N, {item}
    FROM {database}.{schema}.{tbl}
),
TOTAL_EVENTS AS (
    SELECT COUNT(*) AS total_events
    FROM {database}.{schema}.{tbl}
),
ASSOCIATION_RULES AS (
    SELECT
        CNTB.item1 as Antecedent,
        CNTB.item2 as Consequent,
        CNTB.cntb,
        CNT1.cnt1,
        CNT2.cnt2,
        ((cntb * cntb) / (cnt1 * cnt2))::decimal(17,4) AS score,
        100*(cntb / cnt1)::decimal(17,4) AS confidence,
        100*(cntb / N)::decimal(17,4) AS support,
        CASE WHEN (1 - (cntb / cnt1)) = 0 THEN NULL ELSE (1 - (cnt2 / total_events)) / NULLIFZERO(1 - (cntb / cnt1)) END AS conviction,
        (cntb / cnt1) / NULLIFZERO(cnt2 / N) AS lift
    FROM
        CNTB
    JOIN CNT1 ON CNTB.item1 = CNT1.{item}
    JOIN CNT2 ON CNTB.item2 = CNT2.{item}
    JOIN NUMPART ON CNTB.item1 = NUMPART.{item}
    JOIN TOTAL_EVENTS ON 1 = 1
),
LIFT_STATS AS (
    SELECT
        AVG(lift) AS mean_lift,
        STDDEV(lift) AS stddev_lift
    FROM ASSOCIATION_RULES
)
SELECT
ar.*,
(ar.lift - ls.mean_lift) / NULLIF(ls.stddev_lift, 0) AS z_score
FROM
ASSOCIATION_RULES ar,
LIFT_STATS ls;
            """ 
               
    association = session.sql(associationsql).collect()

    #export output as a dataframe
    dfasso = pd.DataFrame(association)
    dfnetwork = pd.DataFrame(association)
    
    with st.expander("Metrics & Filters"):
        st.caption("The association analysis produces association rules (if-then statements) and measures of frequency, relationship, and statistical significance associated with these rules:")
        st.caption("**Antecedent**: represents the 'if' part of the rule, which is an item in the dataset.")
        st.caption("**Consequent**: represents the 'then' part of the rule, which is another item in the dataset.")
        st.caption("**Cntb**: represents the total co-occurences of both antecedent and consequent items.")
        st.caption("**Cnt1**: represents the total occurrences of antecedent item")
        st.caption("**Cnt2**: represents the total occurences of consequent item")
        st.caption("**Score**: represents a calculated value that indicates the strength of association between two items (antecedent and consequent). It helps measure the strength of the association between two items based on how often they co-occur compared to their individual occurrences. ")
        st.caption("**Confidence** (expressed as a percentage): represents the probability that consequent occurs given that antecedent has occurred. A high confidence indicates that when the antecedent is present, the consequent is very likely to also be present.")
        st.caption("**Support** (expressed as a percentage): represents the proportion of transactions that contain both the antecedent and consequent items.")
        st.caption("**Conviction** (expressed as a ratio): measures how strongly the absence of the antecedent implies the absence of the consequent.")
        st.caption("**Lift** (expressed as a ratio): measures how much more likely the consequent is to occur when the antecedent is present compared to if they were independent.")
        st.caption("**Z_Score** (expressed in standard deviations): measures how statistically significant the observed co-occurrence is compared to what would be expected by chance. A higher absolute value of the Z_Score indicates a stronger and more statistically significant association between the items in the rule.")
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            # Selectbox for filtering out a combination of antecedent and consequent (item1 and item2)
            selected_values = st.multiselect("Filter out items", dfasso['ANTECEDENT'].unique())
        
        with col2:
            # Dynamically adjust slider range based on actual values in the DataFrame
            min_score = st.number_input('Minimum Score', 
            min_value=float(dfasso['SCORE'].min()), 
            max_value=float(dfasso['SCORE'].max()), 
            value=float(dfasso['SCORE'].min()))
   
        with col3:
            min_confidence = st.number_input('Minimum Confidence', 
            min_value=float(dfasso['CONFIDENCE'].min()), 
            max_value=float(dfasso['CONFIDENCE'].max()), 
            value=float(dfasso['CONFIDENCE'].min()))
        
        with col4:
            min_support = st.number_input('Minimum Support', 
            min_value=float(dfasso['SUPPORT'].min()), 
            max_value=float(dfasso['SUPPORT'].max()), 
            value=float(dfasso['SUPPORT'].min()))
            
        with col5:
            min_conviction = st.number_input('Minimum Conviction', 
            min_value=float(dfasso['CONVICTION'].min()), 
            max_value=float(dfasso['CONVICTION'].max()), 
            value=float(dfasso['CONVICTION'].min()))
        with col6:
            min_lift = st.number_input('Minimum Lift', 
            min_value=float(dfasso['LIFT'].min()), 
            max_value=float(dfasso['LIFT'].max()), 
            value=float(dfasso['LIFT'].min()))
        with col7:
            min_zscore = st.number_input('Minimum Z_Score', 
            min_value=float(dfasso['Z_SCORE'].min()), 
            max_value=float(dfasso['Z_SCORE'].max()), 
            value=float(dfasso['Z_SCORE'].min()))
     
        # Apply filters to the DataFrame based on the minimum values
        filtered_df = dfasso[
                (dfasso['SCORE'] >= min_score) &
                (dfasso['CONFIDENCE'] >= min_confidence) &
                (dfasso['SUPPORT'] >= min_support) &
                (dfasso['CONVICTION'] >= min_conviction) &
                (dfasso['LIFT'] >= min_lift)&
                (dfasso['Z_SCORE'] >= min_zscore)&
                ~(dfasso['ANTECEDENT'].isin(selected_values) | dfasso['CONSEQUENT'].isin(selected_values))  # Filter out rows where item1 or item2 is in selected_values
            ]
    dfasso = pd.DataFrame(filtered_df)
    if not dfasso.empty:
        # Create two columns for layout
        col1, col2, col3 = st.columns([2, 5,8])
    
        # Place the toggle in the first column
        with col1:
            show_details = st.toggle("Show me!", help='Choose a visualization: Detailed Table or Heatmap or Forced Layout Graph')
    
        # Place the radio button in the second column, but only if the toggle is on
        with col2:
            if show_details:
                genre = st.pills(
                    "Choose a visualization:",
                    ["Detailed Table", "Heatmap", "Graph"],
                    label_visibility="collapsed"
                )
    
        # Place the visualization outside of the columns layout
        if show_details:
            if genre == 'Detailed Table':
                st.dataframe(filtered_df, use_container_width=True)
            elif genre == 'Heatmap':
                # Allow the user to select the metric to display in the heatmap
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    metric = st.selectbox('Choose a metric', dfasso.columns[5:])
                    st.write("")
    
                dfasso[metric] = pd.to_numeric(dfasso[metric], errors='coerce')
                # Create a heatmap
                heatmap_data = dfasso.pivot(index='ANTECEDENT', columns='CONSEQUENT', values=metric)
                n_rows = heatmap_data.shape[0]  # Number of unique ANTECEDENT values (rows)
                n_cols = heatmap_data.shape[1]  # Number of unique CONSEQUENT values (columns)
                font_size = max(5, 20 - max(n_rows, n_cols))
                plt.figure(figsize=(14, 8))
                sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt=".2f", annot_kws={"fontsize": font_size}, cbar=False)
    
                # Set the title and labels
                plt.xlabel("CONSEQUENT", fontsize=12)
                plt.ylabel("ANTECEDENT", fontsize=12)
    
                # Show the plot
                #st.pyplot(plt)
            
                # Create pivot table for heatmap data
                # Step 1: Create heatmap data and round values to 2 decimal places
                all_labels = sorted(set(dfasso['ANTECEDENT']).union(set(dfasso['CONSEQUENT'])))
                heatmap_data = (dfasso.pivot(index='ANTECEDENT', columns='CONSEQUENT', values=metric).reindex(index=all_labels, columns=all_labels).round(2))
                #heatmap_data = dfasso.pivot(index='ANTECEDENT', columns='CONSEQUENT', values=metric).fillna(0).round(2)
                #st.write(heatmap_data)
                #st.write("Unique ANTECEDENTs:", dfasso['ANTECEDENT'].nunique())
                #st.write("Unique CONSEQUENTs:", dfasso['CONSEQUENT'].nunique())
                #st.write("Missing CONSEQUENTs in rows:", set(dfasso['CONSEQUENT'].unique()) - set(dfasso['ANTECEDENT'].unique()))
                # Step 2: Remove diagonal values by setting them to None
             # Step 2: Remove diagonal values by setting them to None
                for idx in heatmap_data.index:
                    if idx in heatmap_data.columns:
                        heatmap_data.at[idx, idx] = None
                # Step 3: Convert to ECharts-compatible list format (skip None values)
                #heatmap_list = [
                #    [i, j, float(heatmap_data.iloc[i, j])]  # Use float to avoid JSON errors
                #    for i in range(heatmap_data.shape[0])
                #    for j in range(heatmap_data.shape[1])
                #    if pd.notna(heatmap_data.iloc[i, j])
                #]
                heatmap_list = [
                    [i, j, float(heatmap_data.iloc[i, j])]
                    for i in range(len(all_labels))
                    for j in range(len(all_labels))
                    if pd.notna(heatmap_data.iloc[i, j])]
                # Step 4: Define options for ECharts heatmap
                options = {
                    "tooltip": {"trigger": "item"},
                    "grid": {
                        "height": "80%",
                        "top": "10%",
                        "left": "10%",
                        "right": "10%",
                        "containLabel": True
                    },
                    "xAxis": {
                        "type": "category",
                        #"data": heatmap_data.columns.tolist(),
                        "data": all_labels,
                        "splitArea": {"show": True},
                        "axisLabel": {
                            "interval": 0,
                            "rotate": 45
                        }
                    },
                    "yAxis": {
                        "type": "category",
                        "data": all_labels,
                        #"data": heatmap_data.index.tolist(),
                        "splitArea": {"show": True}
                    },
                    "visualMap": {
                        "min": heatmap_data.min().min(skipna=True),
                        "max": heatmap_data.max().max(skipna=True),
                        "calculable": True,
                        "orient": "horizontal",
                        "left": "center",
                        "top": "0%",
                        "inRange": {
                            "color": ["#ffffff", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"]
                        },
                        "outOfRange": {
                            "color": "#ffffff"  # White background for 0 or None
                        }
                    },
                    "series": [{
                        "type": "heatmap",
                        "data": heatmap_list,
                        "label": {
                            "show": True,
                        },
                        "emphasis": {
                            "itemStyle": {
                                "shadowBlur": 10,
                                "shadowColor": "rgba(0, 0, 0, 0.5)"
                            }
                        }
                    }]
                }
                # Step 5: Render in Streamlit
                st_echarts(
                    options=options,
                    height="700px",
                    key=f"heatmap_{metric}"
                )
            elif genre == 'Graph':
                # Select a metric for edge color gradient
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    metric = st.selectbox('Choose a metric', dfasso.columns[5:])
                dfasso[metric] = dfasso[metric].astype(float)
                # Visualize the graph
                sigma_graph(dfasso, metric)
    
        if st.toggle("Expl**AI**n Me!", help="Explain Association Rules and derive insights by leveraging Cortex AI..."):
        
    # 1. Define a new function to transform a row of association rule data into text.
            def rule_row_to_text(row):
                """
                Transforms a row of association rule data into a human-readable text string.
                This structured text is ideal for an LLM to understand the rule's meaning and strength.
                """
                # We select the most important metrics for an LLM to interpret a rule's value.
                return (
                    f"Rule: If a user does '{row['ANTECEDENT']}', they are then likely to do '{row['CONSEQUENT']}'. "
                    f"Key Metrics -> "
                    f"Confidence: {row['CONFIDENCE']:.2f}%, "
                    f"Support: {row['SUPPORT']:.2f}%, "
                    f"Lift: {row['LIFT']:.4f}, "
                    f"Z-Score: {row['Z_SCORE']:.4f}."
                )
        
            # 2. Apply the new function to your DataFrame (assumed to be named df_rules).
            # This creates a new column 'RULE_TEXT' containing the formatted string for each rule.
            dfasso['RULE_TEXT'] = dfasso.apply(rule_row_to_text, axis=1)
            
            # 3. Prepare the DataFrame with only the text column to send to Snowpark.
            rules_text_df = dfasso[['RULE_TEXT']]
            
            # --- END OF ADAPTATION ---
        
        
            # The rest of your code remains largely the same, just with updated variable names.
            from snowflake.snowpark.functions import col, lit, call_function
        
            prompt = st.text_input("Prompt your question about the user behavior rules", key="aiassorulesprompt")
            if prompt and prompt != "None":
                try:
                    # Step 1: Create a Snowpark DataFrame from the formatted rule text.
                    rules_snowpark_df = session.create_dataframe(rules_text_df)
                    
                    # Step 2: Apply the AI aggregation directly, using the 'RULE_TEXT' column.
                    ai_result_df = rules_snowpark_df.select(
                        call_function("AI_AGG", col("RULE_TEXT"), lit(prompt)).alias("AI_RESPONSE")
                    )
                    
                    # Step 3: Collect and display the result (this logic is unchanged).
                    explain = ai_result_df.to_pandas()
                    
                    message = st.chat_message("assistant")
                    if not explain.empty:
                        formatted_output = explain.iloc[0, 0]
                        message.markdown(f"**AI Response:**\n\n{formatted_output}")
                    else:
                        message.markdown("No results found for the given prompt.")
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    if st.toggle("Export network data as Gefx", help="Export network data as a Gefx file for further exploration into Gephi Open Graph Platform."):
   
        # Create a network graph
        G = nx.Graph()
    
    # Add nodes
        #nodes = set(dfasso['GAME1']).union(set(dfasso['GAME2']))
        #G.add_nodes_from(nodes)
        for _, row in dfnetwork.iterrows():
            if not G.has_node(row['ANTECEDENT']):
                G.add_node(row['ANTECEDENT'], cnt=row['CNT1'])
        
            if not G.has_node(row['CONSEQUENT']):
                G.add_node(row['CONSEQUENT'], cnt=row['CNT2'])
       
        
    # Add edges with lift as edge weight
        for i, row in dfnetwork.iterrows():
            G.add_edge(row['ANTECEDENT'], row['CONSEQUENT'], cntb=row['CNTB'],cnt1=row['CNT1'],
            score=float(row['SCORE']),
            confidence=float(row['CONFIDENCE']),
            support=float(row['SUPPORT']),
            #conviction=float(row['CONVICTION']),
            lift=float(row['LIFT'] ))
    
    # Draw the graph
        pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)  # For consistent layout
        #plt.figure(figsize=(12, 12))
    
    # Draw edges with thickness representing lift
        edges = nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='black', width=[G[u][v]['cntb'] for u, v in G.edges()])
        
    # Node sizes based on the sum of lift values of connected edges
        #node_size = [0.3*sum([G[u][v]['cnt1'] for u, v in G.nodes(node)]) for node in G.nodes()]
    
    # Draw nodes
        #nx.draw_networkx_nodes(G, pos, node_size=800, node_color="skyblue")
        #nx.draw_networkx_labels(G, pos, font_size=5, font_family="sans-serif")
    
    # Show the plot  
        #st.pyplot(plt)
    
    
    # Save the GEXF data to a BytesIO object
        from io import BytesIO
        output = BytesIO()
        nx.write_gexf(G, output)
        output.seek(0) 
    
    # Provide a download button in Streamlit
        st.download_button(
            label="Download Graph as GEXF",
            data=output,
            file_name="my_network_graph.gexf",
            mime="application/octet-stream")
else:
    st.markdown("""
            <style>
            .custom-container-1 {
                padding: 10px 10px 10px 10px;
                border-radius: 10px;
                background-color: #f0f2f6;  /* Light blue background f0f8ff */
                border: none;  /* Blue border */
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
        
    st.markdown("""
            <div class="custom-container-1">
                <h5 style="font-size: 14px; font-weight: 200 ; color: #29B5E8; margin-top: 0px; margin-bottom: -15px;">
                    Please ensure all required inputs are selected before running the app.
                </h5>
            </div>
            """, unsafe_allow_html=True)
    #st.warning("Please ensure all required inputs are selected before running the app.")
