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
 #SANKEY VIZ
 #--------------------------------------
 # Function for Sankey plot
def sankeyPlot(res, direction, title_text="Sankey nPath", topN=15):
     npath_pandas = res.copy()
     if topN:
         npath_pandas = npath_pandas.sort_values(by='COUNT', ascending=False).head(topN)
 
     if direction == "from":
         dataDict = defaultdict(int)
         for index, row in npath_pandas.iterrows():
             pathCnt = row['COUNT']
             rowList = [item.strip() for item in row['PATH'].split(',')]
             for i in range(len(rowList) - 1):
                 leftValue = rowList[i] + str(i)
                 rightValue = rowList[i + 1] + str(i + 1)
                 valuePair = leftValue + '+' + rightValue
                 dataDict[valuePair] += pathCnt
         
         eventList = []
         for key in dataDict.keys():
             leftValue, rightValue = key.split('+')
             if leftValue not in eventList:
                 eventList.append(leftValue)
             if rightValue not in eventList:
                 eventList.append(rightValue)
         
         sankeyLabel = [s[:-1] for s in eventList]
         sankeySource = []
         sankeyTarget = []
         sankeyValue = []
         
         for key, val in dataDict.items():
             sankeySource.append(eventList.index(key.split('+')[0]))
             sankeyTarget.append(eventList.index(key.split('+')[1]))
             sankeyValue.append(val)
         
         sankeyColor = []
         for i in sankeyLabel:
             sankeyColor.append('#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]))
         
         # Updated Sankey configuration with font properties
         data = go.Sankey(
         arrangement = "snap",
         node = dict(
             pad = 15,
             thickness = 20,
             line = dict(color = "black", width = 0.5),
             label = sankeyLabel,
             color = sankeyColor
         ),
         link = dict(
             source = sankeySource,
             target = sankeyTarget,
             value = sankeyValue,
             color = 'light grey'
         ),
         textfont = dict(
             family = "Arial",
             size = 14,
             color = "black"
         )
     )
         
         fig = go.Figure(data)
         fig.update_layout(
             hovermode='closest',
             title=dict(
                 text=title_text,
                 font=dict(size=20, family="Arial", color="black")
             ),
             plot_bgcolor='white',
             paper_bgcolor='white',
             font=dict(
                 family="Arial",
                 size=14,
                 color="black"
             )
         )
         st.plotly_chart(fig)
 
     elif direction == "to":
         dataDict = defaultdict(int)
         eventDict = defaultdict(int)
         maxPath = npath_pandas['COUNT'].max()
         
         for index, row in npath_pandas.iterrows():
             rowList = row['PATH'].split(',')
             pathCnt = row['COUNT']
             pathLen = len(rowList)
             for i in range(len(rowList) - 1):
                 leftValue = str(150 + i + maxPath - pathLen) + rowList[i].strip()
                 rightValue = str(150 + i + 1 + maxPath - pathLen) + rowList[i + 1].strip()
                 valuePair = leftValue + '+' + rightValue
                 dataDict[valuePair] += pathCnt
                 eventDict[leftValue] += 1
                 eventDict[rightValue] += 1
 
         eventList = []
         for key, val in eventDict.items():
             eventList.append(key)
         sortedEventList = sorted(eventList)
 
         sankeyLabel = []
         for event in sortedEventList:
             sankeyLabel.append(event[3:])
 
         sankeySource = []
         sankeyTarget = []
         sankeyValue = []
         for key, val in dataDict.items():
             sankeySource.append(sortedEventList.index(key.split('+')[0]))
             sankeyTarget.append(sortedEventList.index(key.split('+')[1]))
             sankeyValue.append(val)
 
         sankeyColor = []
         for i in sankeyLabel:
             sankeyColor.append('#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]))
 
         # Updated Sankey configuration with font properties
             data = go.Sankey(
              arrangement = "snap",
         node = dict(
             align="right",
             pad = 15,
             thickness = 20,
             line = dict(color = "black", width = 0.5),
             label = sankeyLabel,
             color = sankeyColor
         ),
         link = dict(
             source = sankeySource,
             target = sankeyTarget,
             value = sankeyValue,
             color = 'light grey'
         ),
         textfont = dict(
             family = "Arial",
             size = 14,
             color = "black"
         )
     )
         
         fig = go.Figure(data)
         fig.update_layout(
             hovermode='closest',
             title=dict(
                 text=title_text,
                 font=dict(size=20, family="Arial", color="black")
             ),
             plot_bgcolor='white',
             paper_bgcolor='white',
             font=dict(
                 family="Arial",
                 size=14,
                 color="black"
             )
         )
         st.plotly_chart(fig)
 
     elif direction == "no_direction":
         dataDict = defaultdict(int)
         for index, row in npath_pandas.iterrows():
             pathCnt = row['COUNT']
             rowList = [item.strip() for item in row['PATH'].split(',')]
             for i in range(len(rowList) - 1):
                 leftValue = rowList[i] + str(i)
                 rightValue = rowList[i + 1] + str(i + 1)
                 valuePair = leftValue + '+' + rightValue
                 dataDict[valuePair] += pathCnt
 
         eventList = []
         for key in dataDict.keys():
             leftValue, rightValue = key.split('+')
             if leftValue not in eventList:
                 eventList.append(leftValue)
             if rightValue not in eventList:
                 eventList.append(rightValue)
 
         sankeyLabel = [s[:-1] for s in eventList]
         sankeySource = []
         sankeyTarget = []
         sankeyValue = []
         for key, val in dataDict.items():
             sankeySource.append(eventList.index(key.split('+')[0]))
             sankeyTarget.append(eventList.index(key.split('+')[1]))
             sankeyValue.append(val)
 
         sankeyColor = []
         for i in sankeyLabel:
             sankeyColor.append('#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]))
 
         # Updated Sankey configuration with font properties
             data = go.Sankey(
         arrangement = "snap",
         node = dict(
             pad = 15,
             thickness = 20,
             line = dict(color = "black", width = 0.5),
             label = sankeyLabel,
             color = sankeyColor
         ),
         link = dict(
             source = sankeySource,
             target = sankeyTarget,
             value = sankeyValue,
             color = 'light grey'
         ),
         textfont = dict(
             family = "Arial",
             size = 14,
             color = "black"
         )
     )
         
         fig = go.Figure(data)
         fig.update_layout(
             hovermode='closest',
             title=dict(
                 text=title_text,
                 font=dict(size=20, family="Arial", color="black")
             ),
             plot_bgcolor='white',
             paper_bgcolor='white',
             font=dict(
                 family="Arial",
                 size=14,
                 color="black"
             )
         )
         st.plotly_chart(fig)
 
     else:
         st.write("Invalid direction.")
 
 #--------------------------------------
 #INTERACTIVE CLICKABLE SANKEY VIZ
 #--------------------------------------
def sankey_chart(df, direction="from",topN_percentage=100):
     dataDict = defaultdict(lambda: {"count": 0, "uids": []})  # Store counts + UIDs
     eventDict = defaultdict(int)
     indexed_paths = []
     dropoffDict = defaultdict(int)  # To store drop-off at each node
     df = df.copy()
 
 
     max_paths = df['PATH'].nunique()
     topN = int(max_paths * (topN_percentage / 100))
 
     if topN:
      df = df.sort_values(by='COUNT', ascending=False).head(topN)
     if direction == "to":
         maxPath = df['COUNT'].max()
         for _, row in df.iterrows():
             rowList = row['PATH'].split(',')
             pathCnt = row['COUNT']
             uid_list = row['UID_LIST']
             pathLen = len(rowList)
             indexedRowList = [f"{150 + i + maxPath - pathLen}_{rowList[i].strip()}" for i in range(len(rowList))]
             indexed_paths.append(",".join(indexedRowList))
             for i in range(len(indexedRowList) - 1):
                 leftValue = indexedRowList[i]
                 rightValue = indexedRowList[i + 1]
                 valuePair = leftValue + '+' + rightValue
                 dataDict[valuePair]["count"] += pathCnt
                 dataDict[valuePair]["uids"].extend(uid_list)
                 eventDict[leftValue] += pathCnt
                 eventDict[rightValue] += pathCnt
         for key, val in dataDict.items():
             source_node = key.split('+')[0]
             target_node = key.split('+')[1]
             tooltip_text = f"""
                 Path: {source_node.split('_', 1)[1]} → {target_node.split('_', 1)[1]}<br>
                 Count: {val["count"]}
             """
             val["tooltip"] = tooltip_text
     elif direction == "from":
         for _, row in df.iterrows():
             rowList = row['PATH'].split(',')
             pathCnt = row['COUNT']
             uid_list = row['UID_LIST']
             indexedRowList = [f"{i}_{rowList[i].strip()}" for i in range(len(rowList))]
             indexed_paths.append(",".join(indexedRowList))
             for i in range(len(indexedRowList) - 1):
                 leftValue = indexedRowList[i]
                 rightValue = indexedRowList[i + 1]
                 valuePair = leftValue + '+' + rightValue
                 dataDict[valuePair]["count"] += pathCnt
                 dataDict[valuePair]["uids"].extend(uid_list)
                 #eventDict[leftValue] += pathCnt
                 eventDict[rightValue] += pathCnt
         # Step 1: Compute drop-offs and forward percentage
         for node in eventDict:
             total_at_node = eventDict[node]
             outgoing = sum(dataDict[f"{node}+{target}"]["count"] for target in eventDict if f"{node}+{target}" in dataDict)
             dropoff = total_at_node - outgoing
             dropoffDict[node] = dropoff
         
         # Step 2: Display drop-offs and forward percentages in the tooltip
         for key, val in dataDict.items():
             source_node = key.split('+')[0]
             target_node = key.split('+')[1]
             total_at_source = eventDict[source_node]
             forward_percentage = (val["count"] / total_at_source * 100) if total_at_source > 0 else 0
             dropoff_percentage = (dropoffDict[source_node] / total_at_source * 100) if total_at_source > 0 else 0
             val["tooltip"] = f"""
                 Path: {source_node.split('_', 1)[1]} → {target_node.split('_', 1)[1]}<br>
                 Count: {val["count"]}<br>
                 Forward %: {forward_percentage:.2f}%<br>
                 Drop-off %: {dropoff_percentage:.2f}%
             """
     sortedEventList = sorted(eventDict.keys())
     sankeyLabel = [event.split('_', 1)[1] for event in sortedEventList]  # Remove index suffix for display
     st.session_state["sankey_labels"] = sankeyLabel
     st.session_state["sortedEventList"] = sortedEventList
     st.session_state["sankey_links"] = dataDict
     sankeySource = []
     sankeyTarget = []
     sankeyValue = []
     sankeyLinks = []
     for key, val in dataDict.items():
         source_node = key.split('+')[0]
         target_node = key.split('+')[1]
         sankeySource.append(sortedEventList.index(source_node))
         sankeyTarget.append(sortedEventList.index(target_node))
         sankeyValue.append(val["count"])
         sankeyLinks.append({
             "source": sortedEventList.index(source_node),
             "target": sortedEventList.index(target_node),
             "value": val["count"],
             "tooltip": {"formatter": val["tooltip"]},
             "uids": val["uids"],
             "source_node": source_node,
             "target_node": target_node
         })
     options = {
         "tooltip": {"trigger": "item"},
         "series": [{
             "type": "sankey",
             "layout": "none",
             "data": [{"label": {"show": True, "formatter": label}} for node, label in zip(sortedEventList, sankeyLabel)],
             "links": sankeyLinks,
             "lineStyle": {"color": "source", "curveness": 0.5},
             "emphasis": {"focus": "adjacency"}
         }]
     }
     return st_echarts(options=options, height="500px", key=f"sankey_chart_{st.session_state['last_df_hash']}", events={"click": "function(params) { return params.data; }"})
 #----------------------
 # TREE VISUALIZATION
 #----------------------    
def build_tree_hierarchy(df, target, direction="to"):
     """
     Builds a hierarchical structure from path data.
     - If direction = "to", treats target as the final event and merges paths leading to it.
     - If direction = "from", treats target as the starting event and builds diverging branches.
     """
     hierarchy = {}
 
     for _, row in df.iterrows():
         path, count, uids = row["PATH"], row["COUNT"], row["UID_LIST"]
         events = path.split(", ")
 
         if target in events:
             if direction == "to":
                 # Target is the final event (reverse the path)
                 target_index = len(events) - 1  
                 processed_events = events[:target_index + 1][::-1]  
             else:
                 # Target is the first event (preserve sequence)
                 target_index = 0  
                 processed_events = events[target_index:]  
 
             # Build hierarchy structure
             current_level = hierarchy
             for event in processed_events:
                 if event not in current_level:
                     current_level[event] = {"children": {}, "count": 0, "uids": set(), "selected": False}  
                 current_level[event]["count"] += count  
                 current_level[event]["uids"].update(uids)  
                 current_level = current_level[event]["children"]  
 
     return hierarchy
 
def convert_to_echarts_format_t(hierarchy, parent_path="", selected_node=None):
     """
     Converts the hierarchy dictionary into ECharts-compatible format.
     Highlights selected node path.
     """
     def recurse(node, path_prefix):
         return [
             {
                 "name": key,
                 "tooltip": {"formatter": f"{key}: {value['count']}"},
                 "lineStyle": {
                     "width": 1 + math.log(value['count'] + 1),
                     "color": "red" if selected_node and selected_node in path_prefix else "#aaa"
                 },
                 "full_path": f"{path_prefix}, {key}".strip(", "),
                 "uids": list(value.get("uids", set())),  
                 "children": recurse(value["children"], f"{path_prefix}, {key}".strip(", "))
             } if value["children"] else {
                 "name": key,
                 "tooltip": {"formatter": f"{key}: {value['count']}"},
                 "lineStyle": {
                     "width": 1 + math.log(value['count'] + 1),
                     "color": "red" if selected_node and selected_node in path_prefix else "#aaa"
                 },
                 "full_path": f"{path_prefix}, {key}".strip(", "),
                 "uids": list(value.get("uids", set()))  
             }
             for key, value in node.items()
         ]
     return recurse(hierarchy, parent_path)
     
 
def plot_tree(df, target, direction="to", selected_node=None):
     hierarchy = build_tree_hierarchy(df, target, direction)
     echarts_data = convert_to_echarts_format_t(hierarchy, selected_node=selected_node)
 
     # Set tree orientation based on direction
     orient = "RL" if direction == "to" else "LR"  
     label_position = "left" if direction == "to" else "right"
     align = "right" if direction == "to" else "left"
 
     tree_options = {
         "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
         "series": [
             {
                 "type": "tree",
                 "data": echarts_data,
                 "top": "5%",
                 "left": "5%",
                 "bottom": "5%",
                 "right": "20%",
                 "symbolSize": 15,
                 "orient": orient,
                 "label": {
                     "position": label_position,
                     "verticalAlign": "middle",
                     "align": align,
                     "fontSize": 10
                 },
                 "leaves": {
                     "label": {
                         "position": "right" if direction == "to" else "left",
                         "verticalAlign": "middle",
                         "align": "left" if direction == "to" else "right",
                         "fontSize": 10,
                     }
                 },
                 "expandAndCollapse": True,
                 "initialTreeDepth": -1,
                 "animationDuration": 550,
                 "animationDurationUpdate": 750,
                 "lineStyle": {
                     "curveness": 0.6,
                     "color": "#aaa"  
                 },
                 "layout": "orthogonal",
                 "nodePadding": 40,
                 "depth": 6
             }
         ]
     }
 
     events = {
         "click": "function(params) { return {full_path: params.data.full_path, uids: params.data.uids}; }"
     }
     
     clicked_node = st_echarts(options=tree_options, events=events, height="800px", width="100%")
 
     return clicked_node
     
 #---------------------------
 # SIGMA VIZ
 #---------------------------
def rgba_to_str(rgba):
  return f"rgba({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)}, {rgba[3]})"
def sigma_graph(df):
 # Step 1: Extract event pairs and calculate weights
     event_pairs = []
     event_counts = Counter()
     for _, row in df.iterrows():
         path_str, weight = row["PATH"], row["COUNT"]
         events = path_str.split(", ")
         pairs = [(events[i], events[i + 1]) for i in range(len(events) - 1)]
         for pair in pairs:
             event_pairs.extend([pair] * weight)
         for event in events:
             event_counts[event] += weight
 
     pair_counts = Counter(event_pairs)
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
     
     max_pair_count = max(pair_counts.values())
     norm = Normalize(vmin=0, vmax=max_pair_count)
     log_counts = {pair: np.log1p(count) for pair, count in pair_counts.items()}
     log_max = max(log_counts.values())
     max_width = 6
 
     edges = [
         {
             "source": src,
             "target": tgt,
             "lineStyle": {"width":min(np.log(count + 1), max_width) , "color": rgba_to_str(plt.cm.Blues(norm(count)))},  # Logarithmic scaling for edge thickness   # Color based on count
             "tooltip": {
                 "formatter": f"{src} → {tgt}<br>Count: {count}<br>Percentage: {count / total_pairs:.2%}"
             },
         }
         for (src, tgt), count in pair_counts.items()
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
 #--------------------
 #SUNBURST
 #--------------------
 # Function to generate random colors for each event
def generate_colors(events):
     color_map = {}
     for event in events:
         color_map[event] = '#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
     return color_map
# Build a hierarchical structure based on direction
def build_hierarchy(paths, direction="from"):
    if direction == "to":
        # Reverse the order of the path for "to" visualization
        paths = [(path.split(", ")[::-1], size) for path, size in paths]
    else:
        paths = [(path.split(", "), size) for path, size in paths]
    root = {"name": "", "children": [], "path": [], "value": 0}
    
    for parts, size in paths:
        current_node = root
        current_path = []
        for node_name in parts:
            current_path.append(node_name)
            children = current_node.setdefault("children", [])
            found_child = None
            
            for child in children:
                if child["name"] == node_name:
                    found_child = child
                    break
            
            if found_child:
                current_node = found_child
            else:
                new_node = {
                    "name": node_name,
                    "children": [],
                    "path": current_path.copy(),
                    "value": 0
                }
                children.append(new_node)
                current_node = new_node
            
            current_node["value"] += size
    # ✅ If direction is "to", reverse order of children to reflect "to" logic
    if direction == "to":
        root["children"] = root["children"][::-1]
    return root
# Convert hierarchy to ECharts-compatible format
def convert_to_echarts_format(hierarchy, color_map, total_count, direction="from"):
    def recurse(node):
        color = color_map.get(node["name"], "#FF5722")  # Default color if not in the map
        percentage = f"{(node.get('value', 0) / total_count * 100):.2f}%" if node.get("value") else ""
        full_path = " → ".join(node.get("path", []))
        
        # ✅ Fix tooltip direction based on the `direction` parameter
        if direction == "to":
            #full_path = " ← ".join(reversed(node.get("path", [])))
            full_path = " → ".join(reversed(node.get("path", [])))
        
        return {
            "name": node["name"],
            "value": node.get("value", None),
            "tooltip": {
                "formatter": f"Path: {full_path}<br>Count: {node.get('value', 0)}<br>Percentage: {percentage}"
            },
            "itemStyle": {
                "color": color
            },
            "children": [recurse(child) for child in node.get("children", [])] if "children" in node else None
        }
    
    return [recurse(hierarchy)]
# Extract events and counts from DataFrame
def extract_events_and_count(df):
    global total_count, events
    events = set()
    total_count = 0
    for _, row in df.iterrows():
        path = row['PATH']
        size = row['COUNT']
        path_events = path.split(", ")
        for event in path_events:
            events.add(event)
        total_count += size
#Main function to generate Sunburst chart
def process_and_generate_sunburst(df, direction="to"):
    extract_events_and_count(df)
    
    # Generate consistent colors for each event
    color_map = generate_colors(events)
    
    #Build hierarchy with the correct path order
    paths = [(row['PATH'], row['COUNT']) for _, row in df.iterrows()]
    hierarchy = build_hierarchy(paths, direction)
    
    #Convert hierarchy to ECharts format
    echarts_data = convert_to_echarts_format(hierarchy, color_map, total_count, direction)[0]["children"]
    
    #Sunburst chart configuration
    sunburst_options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "{b}: {c}", 
            "position": [0, 20], 
            "extraCssText": "z-index: 10;"
        },
        "series": [
            {
                "type": "sunburst",
                "data": echarts_data,
                "radius": ["20%", "90%"],
                "label": {
                    "show": False
                },
                "itemStyle": {
                    "borderColor": "#fff",
                    "borderWidth": 1
                },
                "highlightPolicy": "ancestor",
            }
        ]
    }
    
    # Display the Sunburst chart
    st_echarts(options=sunburst_options, height="600px")
    #Legend configuration
    legend_columns = st.columns(3)  
    col_idx = 0
    for event, color in color_map.items():
        with legend_columns[col_idx]:
            st.markdown(f"<span style='color:{color};font-weight:normal'>■</span> {event}", unsafe_allow_html=True)
        
        col_idx = (col_idx + 1) % 3
        # Get the current credentials
session = get_active_session()
#--------------------------------------
#VAR INIT
#--------------------------------------
#Initialize variables
fromevt = None
fromevt1=None
toevt = None
toevt1=None
minnbbevt = 0
minnbbevt1 = 0
maxnbbevt = 5
maxnbbevt1 = 5
overlap = 'PAST LAST ROW'
overlap1 = 'PAST LAST ROW'
uid = None
evt = None
display = None
tmstp = None
tbl = None
partitionby = None
groupby = None
startdt_input = None
enddt_input = None
sess = None
uid1 = None
evt1 = None
tmstp1 = None
tbl1 = None
partitionby1 = None
groupby1 = None
startdt_input1 = None
enddt_input1 = None
sess1 = None
excl3_instance = "''"
timeout = None
unitoftime= None
timeout1 = None
unitoftime1= None
cols = ''
cols1=''
colsdf = pd.DataFrame()
colsdf1 = pd.DataFrame()
#--------------------------------------
#PAGE TITLE
#--------------------------------------
st.markdown("""
    <style>
    .custom-container-1 {
        padding: 10px 10px 10px 10px;
        border-radius: 10px;
        background-color: #f0f2f6;  /* Light blue background f0f8ff */
        border: 1px solid #29B5E8;  /* Blue border */
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="custom-container-1">
        <h5 style="font-size: 18px; font-weight: normal; color: #29B5E8; margin-top: 0px; margin-bottom: -15px;">
            PATH ANALYSIS
        </h5>
    </div>
    """, unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Analyze", "Compare"])


#--------------------------------------
#ANALYZE TAB
#--------------------------------------
with tab1:
    with st.expander("Input Parameters"):
            
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
                database = st.selectbox('Select Database', key='analyzedb', index=None, 
                                        placeholder="Choose from list...", options=db0['name'].unique())
            
            # **Schema Selection (Only if a database is selected)**
            if database:
                sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                schemas = session.sql(sqlschemas).collect()
                schema0 = pd.DataFrame(schemas)
            
                with col2:
                    schema = st.selectbox('Select Schema', key='analyzesch', index=None, 
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
                    tbl = st.selectbox('Select Event Table or View', key='analyzetbl', index=None, 
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

            col1, col2, col3 = st.columns([4,4,4])
            with col1:
                uid = st.selectbox('Select identifier column', colsdf, index=None, placeholder="Choose from list...", help="The identifier column is the unique identifier for each event record. This may be a user id, a customer id, a process id, etc ...it is used to partition the input set of rows before pattern matching.")
            with col2: 
                evt = st.selectbox('Select event column', colsdf, index=None, placeholder="Choose from list...",help="The event column contains the actual events that will be used in the path analysis.")
            with col3:
                tmstp = st.selectbox('Select timestamp column', colsdf, index=None, placeholder="Choose from list...",help="The timestamp column contains the timestamp associated to the event. This is the value that will be used for sequentially ordering the dataset.")
                        
                
            checkdisplay = st.toggle ("Display Column", key="checkdisplaycolumn",help="When toggled on, you can select an alternative column (such as channel, campaign, etc.) to be displayed in the visualizations and used by the AI-powered explanations. This enables you to view and interpret user paths at a higher level of abstraction, while the event column remains the basis for the underlying path analysis. When toggled off, the event column is used consistently for pattern definition, visual display, and AI-powered explanations.")
            display=evt
            if checkdisplay:

                col1, col2 = st.columns([4,8])
                with col1 :
                    selecteddisplay = st.selectbox('Select display column', colsdf, index=None, placeholder="Choose from list...",help="The column that will be used in the path diagrams.")
                    if selecteddisplay is not None:
                        display = selecteddisplay
                with col2:
                    st.write("")
            else: display = evt
                
            #Get Distinct Events Of Interest from Event Table
            if (uid != None and evt != None and tmstp != None):
            # Get Distinct Events Of Interest from Event Table
                EOI = f"SELECT DISTINCT {evt} FROM {tbl} ORDER BY {evt}"
                        # Get start EOI :
                start = session.sql(EOI).collect()
            # Get end EOI :
                end = session.sql(EOI).collect()
            # Get excluded EOI :
                excl =session.sql(EOI).collect()
        
                # Write query output in a pandas dataframe
                startdf0 = pd.DataFrame(start)
                enddf0 = pd.DataFrame(end)
                excl0 =pd.DataFrame(excl)   


            #Add "any" to the distinct events list
                any_row = {evt: 'Any'}

        # Convert the any_row to a DataFrame and append it using pd.concat
                any_row_df = pd.DataFrame([any_row])
        # Add any to list of FROM 
                startdf1 = pd.concat([ any_row_df, startdf0], ignore_index=True)
        # Add any to list of TO
                enddf1 = pd.concat([ any_row_df, enddf0], ignore_index=True)

        #EVENTS PATTERN
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Events Pattern</h2>
        <hr style='margin-top: -8px;margin-bottom: 5px;'>
        """, unsafe_allow_html=True)
                                
                # Add a None placeholder to force user to select an event
                options_with_placeholder_from = ["🔍"] + startdf1[evt].unique().tolist()
                options_with_placeholder_to = ["🔎"] + enddf1[evt].unique().tolist()
                    
                col1, col2, col3,col4 = st.columns([4,4,2,2])

                with col1:
                    frm = st.multiselect('Select events FROM:', options=options_with_placeholder_from[1:], default=[],help="Select one or more events of interest to visualize paths FROM the selected point(s). 'Any' matches all values.")
                    #filtered_frmevt = startdf1[(startdf1[evt] == frm)]
                    #fromevt = filtered_frmevt.iloc[0, 0]
                    if frm != "🔍":
                        fromevt= ", ".join([f"'{value}'" for value in frm])
 
                    else:
                        fromevt = None  # Set to None if the placeholder is still selected
               
                with col2:
                    to = st.multiselect('Select events TO:', options=options_with_placeholder_to[1:], default=[],help="Select one or more events of interest to visualize paths TO the selected point(s). 'Any' matches all values.")
                    #filtered_toevt = enddf1[(enddf1[evt] == to)]
                    #toevt =filtered_toevt.iloc[0, 0]
                    if to != "🔎":
                        toevt = ", ".join([f"'{value}'" for value in to])
    
                    else:
                        toevt = None  # Set to None if the placeholder is still selected
                
                        
                with col3:
                    minnbbevt = st.number_input("Min # events", value=0, placeholder="Type a number...",help="Select the minimum number of events either preceding or following the event(s) of interest.")
                with col4:
                    maxnbbevt = st.number_input("Max # events", value=5, min_value=1, placeholder="Type a number...",help="Select the maximum number of events either preceding or following the event(s) of interest.")
               
                col1, col2 = st.columns([5,10])
                with col1:
                    checkoverlap = st.toggle("Allow overlap", key="checkoverlapanalyze",help="Specifies the pattern-matching mode. Toggled-on allows OVERLAP and finds every occurrence of pattern in partition, regardless of whether it is part of a previously found match. One row can match multiple symbols in a given matched pattern.Toggled-off does not allow OVERLAP and starts next pattern search at row that follows last pattern match.")
                    overlap = 'PAST LAST ROW'
                if checkoverlap:
                    overlap='TO NEXT ROW'
                with col2:
                    st.write("")    
        
                
    #--------------------------------------
    #DATE RANGE
    #--------------------------------------
                with st.container():
                    st.markdown("""
                    <h2 style='font-size: 14px; margin-bottom: 0px;'>Date range</h2>
                    <hr style='margin-top: -8px;margin-bottom: 5px;'>
                    """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns([2.4,2.4,10])

                # SQL query to get the min start date
                minstartdt = f"SELECT   TO_VARCHAR(MIN ({tmstp}), 'YYYY/MM/DD') FROM {tbl}"
                # Get min start date :
                defstartdt = session.sql(minstartdt).collect()
                defstartdt_str = defstartdt[0][0] 
                defstartdt1 = datetime.datetime.strptime(defstartdt_str, '%Y/%m/%d').date()
                
                with col1:
                    startdt_input = st.date_input('Start date', value=defstartdt1)
                with col2:
                    enddt_input = st.date_input('End date',help="Apply a time window to the data set to find paths on a specific date range or over the entire lifespan of your data (default values)")
                with col3:
                    st.write("")
                    
    #--------------------------------------
    #SESSION
    #--------------------------------------
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Session (optional)</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True,help= "A session, in the context of customer journey analysis, refers to a defined period of activity during which a customer interacts or events occur. It includes all events, actions, or transactions performed by the user within this timeframe. If a session field is present in the event table, select it from the 'Column' tab below. If no session field is available, use the 'Sessionize' tab to create unique session identifiers by grouping events based on time gaps (e.g., a gap of more than 30 minutes starts a new session). Once a session is selected or created, it will be used alongside the unique identifier to partition the input rows before applying pattern matching.")
                tab11, tab22 = st.tabs(["Column", "Sessionize"])
               
                with tab11:
                    col1, col2  = st.columns([5,10])
                    with col1:
                            sess = st.selectbox('Select session column ', colsdf, index=None, placeholder="Choose from list...",help="If a session field is available within the event table.")
                    
                    with col2:
                        st.write("")
                
                with tab22:
                    col1, col2, col3 = st.columns([2.4,2.4,10])
                    with col1:
                        unitoftime =st.selectbox( "Unit of time",
                    ("SECOND", "MINUTE", "HOUR","DAY"),index=None, placeholder="Choose from list", help="Select the unit of time of the session time window.")
                  
                    with col2:
                        timeout=  st.number_input( "Insert a timeout value",value=None, min_value=1, format="%d", placeholder="Type a number",help="Value of the session time window.")
                if sess == None and unitoftime==None and timeout==None: 
                        partitionby = f"partition by {uid}"
                        groupby = f"group by {uid}, match_number "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp])]['COLUMN_NAME']
                
                elif sess != None and unitoftime==None and timeout==None:
                        partitionby=f"partition by {uid},{sess} "
                        groupby = f"group by {uid}, match_number,{sess} "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp,sess])]['COLUMN_NAME']
    
                elif sess == None and unitoftime !=None and timeout !=None:
                        partitionby=f"partition by {uid},SESSION "
                        groupby = f"group by {uid}, match_number,SESSION "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp])]['COLUMN_NAME']
                else :
                    st.write("")
                    
                    
    #--------------------------------------
    #FILTERS
    #--------------------------------------

                #initialize sql_where_clause
                sql_where_clause = ""

                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Filters</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)

                    col1, col2, col3  =st.columns([5,2,8])
                    with col1:
                # Exclude Events
                        excl1 = st.multiselect('Exclude event(s) - optional', excl0,placeholder="Select event(s)...",help="Event(s) to be excluded from the pattern evaluation and the ouput.") 

                        if not excl1:
                          excl3 = "''"
                        else:
                         excl3= ', '.join([f"'{excl2}'" for excl2 in excl1])
                    with col2:
                        # Top n paths
                        topn = st.number_input("Top paths to show", value=100, placeholder="Type a number...",help="Shows the most frequent paths")
                    with col3:
                        st.write("")
                
                # Ensure we have remaining columns before allowing additional filters
                if not remaining_columns.empty:
                    col1, col2 = st.columns([5, 10])
                
                    with col1:
                        checkfilters = st.toggle("Additional filters", help="Apply one or many conditional filters to the input data used in the path and pattern analysis.")
                    with col2:
                        st.write("")    
                else:
                    checkfilters = False  # Disable filters when no columns are left
                
                # Only execute filter logic if there are available columns AND the toggle is enabled
                if checkfilters and not remaining_columns.empty:
                
                    with st.container():
                        # Helper function to fetch distinct values from a column
                        def fetch_distinct_values(column):
                            """Query the distinct values for a column, except for dates"""
                            query = f"SELECT DISTINCT {column} FROM {tbl}"
                            result = session.sql(query).collect()
                            distinct_values = [row[column] for row in result]
                            return distinct_values
                
                        # Helper function to display operator selection based on column data type
                        def get_operator_input(col_name, col_data_type, filter_index):
                            """ Returns the operator for filtering based on column type """
                            operator_key = f"{col_name}_operator_{filter_index}"  # Ensure unique key
                
                            if col_data_type in ['NUMBER', 'FLOAT', 'INT']:
                                operator = st.selectbox("Operator", ['=', '<', '<=', '>', '>=', '!=', 'IN'], key=operator_key)
                            elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                                operator = st.selectbox("Operator", ['<=', '>=', '='], key=operator_key)
                            else:  # For string or categorical columns
                                operator = st.selectbox("Operator", ['=', '!=', 'IN'], key=operator_key)
                            return operator
                
                        # Helper function to display value input based on column data type
                        def get_value_input(col_name, col_data_type, operator, filter_index):
                            """ Returns the value for filtering based on column type """
                            value_key = f"{col_name}_value_{filter_index}"  # Ensure unique key
                
                            if operator == 'IN':
                                distinct_values = fetch_distinct_values(col_name)
                                value = st.multiselect(f"Values for {col_name}", distinct_values, key=value_key)
                            elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                                value = st.date_input(f"Value for {col_name}", key=value_key)
                            else:
                                distinct_values = fetch_distinct_values(col_name)
                                value = st.selectbox(f"Value for {col_name}", distinct_values, key=value_key)
                
                            return value
                
                        # Initialize variables to store filters and logical conditions
                        filters = []
                        logic_operator = None
                        filter_index = 0
                
                        while True:
                            available_columns = remaining_columns  # Columns available for filtering
                
                            if available_columns.empty:
                                st.write("No more columns available for filtering.")
                                break  # Stop the loop if no columns remain
                
                            # Create 3 columns for column selection, operator, and value input
                            col1, col2, col3 = st.columns([2, 1, 2])
                
                            with col1:
                                selected_column = st.selectbox(f"Column (filter {filter_index + 1})", available_columns)
                
                            # Determine column data type
                            column_info_query = f"""
                                SELECT DATA_TYPE
                                FROM INFORMATION_SCHEMA.COLUMNS
                                WHERE TABLE_NAME = '{tbl}' AND COLUMN_NAME = '{selected_column}';
                            """
                            column_info = session.sql(column_info_query).collect()
                            col_data_type = column_info[0]['DATA_TYPE']  # Get the data type
                
                            with col2:
                                operator = get_operator_input(selected_column, col_data_type, filter_index)
                
                            with col3:
                                value = get_value_input(selected_column, col_data_type, operator, filter_index)
                
                            # Append filter if valid
                            if operator and (value is not None or value == 0):
                                filters.append((selected_column, operator, value))
                
                            # Ask user if they want another filter
                            add_filter = st.radio(f"Add another filter after {selected_column}?", ['No', 'Yes'], key=f"add_filter_{filter_index}")
                
                            if add_filter == 'Yes':
                                col1, col2 = st.columns([2, 13])
                                with col1: 
                                    logic_operator = st.selectbox(f"Choose logical operator after filter {filter_index + 1}", ['AND', 'OR'], key=f"logic_operator_{filter_index}")
                                    filter_index += 1
                                with col2:
                                    st.write("")
                            else:
                                break
                        
                        # Generate SQL WHERE clause based on selected filters and logic
                        sql_where_clause = " AND "
                        #st.write(filters)
                        for i, (col, operator, value) in enumerate(filters):
                            if i > 0 and logic_operator:
                                sql_where_clause += f" {logic_operator} "
                            
                            if operator == 'IN':
                                # Handle IN operator where the value is a list
                                sql_where_clause += f"{col} IN {tuple(value)}"
                            else:
                                # Check if value is numeric
                                    if isinstance(value, (int, float)):
                                        sql_where_clause += f"{col} {operator} {value}"
                                    else:
                                 # For non-numeric values (strings, dates), enclose the value in quotes
                                        sql_where_clause += f"{col} {operator} '{value}'"        
                        else:
                            st.write("")
        
    # SQL LOGIC
    # Check pattern an run SQL accordingly
            

    if all([uid, evt, tmstp,fromevt, toevt]):
        # Now we can proceed with the SQL logic and any further processing
        
        # Continue with SQL generation and execution based on inputs...

        # PATH TO: Pattern = A{{{minnbbevt},{maxnbbevt}}} B
        if fromevt.strip("'") == 'Any' and toevt.strip("'") != 'Any':
            
            path_to_agg=None
            path_to_det_df=None
            path_to_det_sql=None
            path_to_agg_sql= None
            # Aggregate results for plot
            if unitoftime==None and timeout ==None :
                
                path_to_agg_sql = f"""
                select top {topn} path, count(*) as count,array_agg({uid}) as uid_list from (
                    select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                        from (select * from {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                            match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A{{{minnbbevt},{maxnbbevt}}} B)
                            define A as true, B AS {evt} IN({toevt}))
                    {groupby} ) 
                group by path order by count 
                """
    
                path_to_agg = session.sql(path_to_agg_sql).collect()
            elif unitoftime != None and timeout !=None :
                path_to_agg_sql = f"""
            select top {topn} path, count(*) as count,array_agg({uid}) as uid_list from (
                select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                    from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                    {tmstp}) AS TIMEWINDOW
                 FROM
                {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
         ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session
        FROM events_with_diff)
        SELECT *FROM sessions)
                        match_recognize(
                        {partitionby} 
                        order by {tmstp}  
                        measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                        all rows per match
                        AFTER MATCH SKIP {overlap} 
                        pattern(A{{{minnbbevt},{maxnbbevt}}} B)
                        define A as true, B AS {evt} IN({toevt}))
                {groupby} ) 
            group by path order by count 
            """
            
                path_to_agg = session.sql(path_to_agg_sql).collect()
            # If the DataFrame is not empty, show Sankey plot
            
            res = pd.DataFrame(path_to_agg)
            #st.write (res)
            import ast
        
                # Print to verify
            #st.write("✅ Processed DataFrame:", res)
                            
            if not res.empty:
                def convert_uid_list(uid_entry):
                    if isinstance(uid_entry, str):
                        try:
                            return ast.literal_eval(uid_entry)  # Safely convert string to list
                        except (SyntaxError, ValueError):
                            return []  # Return empty list if conversion fails
                    return uid_entry  # Already a list, return as is
                
                # Apply the function ONLY to UID_LIST while keeping other columns unchanged
                res["UID_LIST"] = res["UID_LIST"].apply(convert_uid_list)
                # Create two columns for layout
                col1, col2,col3 = st.columns([2,7,6])
            
                # Place the toggle in the first column
                with col1:
                    show_details = st.toggle("Show me!", help="Select a visualization option: Sankey, Tree, Forced Layout Graph or Sunburst.")
                    #show_details = st.toggle("Show me!", help="Select a visualization option: Sankey, Tree, Forced Layout Graph or Sunburst. In the Tree diagram you can click on a node to select a specific path. Once a path is selected, it will be displayed, showing the sequence of events leading to or from the chosen point. To create a customer segment, expand the CREATE SEGMENT section, choose a database and schema from the available options and enter a segment table name where the user IDs will be stored. After configuring these details, create a new segment by clicking 'Create Segment' which will generate a new table and insert the selected user IDs, or append the selected IDs to an existing table by clicking 'Append to Segment'.")


                # Place the radio button in the second column, but only if the toggle is on
                with col2:
                    if show_details:
                        genre = st.pills(
                            "Choose a visualization:",
                            ["Sankey","Tree", "Graph", "Sunburst"],
                            label_visibility="collapsed"
                        )
            
                # Place the visualization outside of the columns layout
                if show_details:
                    if genre == 'Sankey':
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            percentage = st.slider("Display Top % of Paths", 1, 100, 100)
                        with col2:
                            st.write("")
                      
                        # STEP 2: Reset Session State on Query Change**
                        current_df_hash = hash(res.to_json())
                        if "last_df_hash" not in st.session_state or st.session_state["last_df_hash"] != current_df_hash:
                            st.session_state["last_df_hash"] = current_df_hash  # Update hash
                            # Reset state variables
                            st.session_state["clicked_sankey"] = None
                            st.session_state["clicked_source"] = None
                            st.session_state["clicked_target"] = None
                            st.session_state["selected_uids"] = set()
                            st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                            st.session_state["sankey_chart"] = None
                            st.session_state["sankey_links"] = {}
                            st.session_state["sankey_labels"] = []
                            st.session_state["sortedEventList"] = []
                            
                            #st.info("ℹ️ **Query executed! Resetting selections & memory.**")
                        # Session state to store extracted user data
                        if "selected_paths_df" not in st.session_state:
                            st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                        
                        clicked_sankey = sankey_chart(res, direction="to", topN_percentage=percentage)
                        
                        if clicked_sankey:
                            sankeyLabel = st.session_state.get("sankey_labels", [])
                            sortedEventList = st.session_state.get("sortedEventList", [])
                            sankeyLinks = st.session_state.get("sankey_links", {})
                            if "source" in clicked_sankey and "target" in clicked_sankey:
                                source_index = clicked_sankey["source"]
                                target_index = clicked_sankey["target"]
                                clicked_source = sortedEventList[source_index]
                                clicked_target = sortedEventList[target_index]
                                st.caption(f"Selected Edge: {clicked_source.split('_', 1)[1]} → {clicked_target.split('_', 1)[1]}")
                                valuePair = f"{clicked_source}+{clicked_target}"
                                extracted_uids = sankeyLinks.get(valuePair, {}).get("uids", [])
                                #st.write(f"👤 Extracted UIDs: {extracted_uids}")
                                flattened_uids = set()
                                for uid_list in extracted_uids:
                                    if isinstance(uid_list, list):
                                        flattened_uids.update(map(str, uid_list))
                                    else:
                                        flattened_uids.add(str(uid_list))
                                # Store in session state (initialize if not present)
                                if "selected_uids" not in st.session_state:
                                    st.session_state["selected_uids"] = flattened_uids
                                else:
                                    st.session_state["selected_uids"].update(flattened_uids)  # Accumulate UIDs
                                # Update DataFrame for UI display
                                new_entry = pd.DataFrame([{"Source": clicked_source, "Target": clicked_target, "User_IDs": list(flattened_uids)}])
                                if "selected_paths_df" not in st.session_state:
                                    st.session_state["selected_paths_df"] = new_entry
                                else:
                                    st.session_state["selected_paths_df"] = pd.concat(
                                        [st.session_state["selected_paths_df"], new_entry], ignore_index=True
                                    )
                                #st.write(f"👤 Extracted UIDs: {list(flattened_uids)}")
                        #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        # 🔹 : Add a Manual Reset Button**
                        # if st.button("🔄 Reset Selection", use_container_width=False):
                        #     # Reset all relevant session state variables
                        #     st.session_state["selected_uids"] = set()
                        #     st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                        #     st.session_state["clicked_sankey"] = None
                        #     st.session_state["clicked_source"] = None
                        #     st.session_state["clicked_target"] = None
                        #     st.session_state["sankey_links"] = {}
                        #     st.session_state["sankey_labels"] = []
                        #     st.session_state["sortedEventList"] = []
                            
                        #     st.session_state["refresh_key"] = not st.session_state.get("refresh_key", False)
                        #     #st.success("✅ Selections & memory cleared!")
                        #     #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        
                        # with st.expander("CREATE SEGMENT"):
                        #     # Fetch available databases
                        #     sqldb = "SHOW DATABASES"
                        #     databases = session.sql(sqldb).collect()
                        #     db0 = pd.DataFrame(databases)
                        #     col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        #     # Database Selection
                        #     with col1:
                        #         database = st.selectbox("Select Database", key="segmentdb", index=None, 
                        #                                 placeholder="Choose from list...", options=db0["name"].unique())
                        #     # Schema Selection
                        #     schema = None
                        #     if database:
                        #         sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                        #         schemas = session.sql(sqlschemas).collect()
                        #         schema0 = pd.DataFrame(schemas)
                        #         with col2:
                        #             schema = st.selectbox("Select Schema", key="segmentsch", index=None, 
                        #                                 placeholder="Choose from list...", options=schema0["name"].unique())
                        #     # Table Name Input
                        #     table_name = None
                        #     if database and schema:
                        #         with col3:
                        #             table_name = st.text_input("Enter Segment Table Name", key="segmenttbl", placeholder="Type table name...")
                        #     # ✅ Use selected UIDs directly from session_state
                        #     selected_uids = st.session_state.get("selected_uids", set())
                        #     if database and schema and table_name and selected_uids:
                        #         #st.write(f"🔍 Extracted UIDs for insertion: {selected_uids}")
                        #         create_table_sql = f"""
                        #         CREATE OR REPLACE TABLE {database}.{schema}.{table_name} (
                        #             ID STRING
                        #         )
                        #         """
                        #         # Convert UIDs to SQL-safe format
                        #         values = ", ".join([f"('{uid}')" for uid in selected_uids])
                        #         insert_sql = f"INSERT INTO {database}.{schema}.{table_name} (ID) VALUES {values};" if values else ""
                        #         #st.write(f"🔍 SQL Insert Preview:\n{insert_sql}")  # Debugging
                        #         with col4:
                        #             if st.button("Create Segment", use_container_width=True):
                        #                 try:
                        #                     session.sql(create_table_sql).collect()
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                     st.success(f"✅ Segment `{database}.{schema}.{table_name}` created successfully!")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
                        #         with col5:
                        #             if st.button("Append to Segment", use_container_width=True):
                        #                 try:
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                         st.success(f"✅ UIDs successfully appended to `{database}.{schema}.{table_name}`!")
                        #                     else:
                        #                         st.warning("⚠️ No UIDs selected to append.")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
                    elif genre == 'Tree':
                        target_event = toevt.strip("'")
                        clicked_node= plot_tree(res, target_event, "to")
                        if clicked_node:
                            selected_path = clicked_node["full_path"]  
                            selected_uids = clicked_node["uids"]  # Directly use cleaned UIDs
                        
                            direction = "to"
                            if direction == "to":
                                selected_path = ", ".join(reversed(selected_path.split(", ")))  # Reverse only in "to" mode
                        
                            st.write(f"🔀 Selected path: **{selected_path}**")
                            
                            with st.expander("CREATE SEGMENT"):  # Keep the form visible
                                # **Fetch available databases**
                                sqldb = "SHOW DATABASES"
                                databases = session.sql(sqldb).collect()
                                db0 = pd.DataFrame(databases)
                        
                                # **Row Layout: Database, Schema, Table Name, Create & Append Buttons**
                                col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        
                                # **Database Selection**
                                with col1:
                                    database = st.selectbox("Select Database", key="segmentdb", index=None, 
                                                            placeholder="Choose from list...", options=db0["name"].unique())
                        
                                # **Schema Selection**
                                schema = None
                                if database:
                                    sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                                    schemas = session.sql(sqlschemas).collect()
                                    schema0 = pd.DataFrame(schemas)
                        
                                    with col2:
                                        schema = st.selectbox("Select Schema", key="segmentsch", index=None, 
                                                              placeholder="Choose from list...", options=schema0["name"].unique())
                        
                                # **Table Name Input**
                                table_name = None
                                if database and schema:
                                    with col3:
                                        table_name = st.text_input("Enter Segment Table Name", key="segmenttbl", placeholder="Type table name...")
                        
                                # **Ensure all fields are filled before proceeding**
                                if database and schema and table_name and selected_uids:
                                    # ✅ **STEP 1: GENERATE SQL STATEMENTS**
                                    create_table_sql = f"""
                                    CREATE OR REPLACE TABLE {database}.{schema}.{table_name} (
                                        ID STRING
                                    )
                                    """
                        
                                    values = ", ".join([f"('{uid}')" for uid in selected_uids])  # Format UIDs
                                    insert_sql = f"INSERT INTO {database}.{schema}.{table_name} (ID) VALUES {values};"
                        
                                    # ✅ **STEP 2: SHOW SQL PREVIEW**
                                    #st.write("#### SQL Preview:")
                                    #st.code(create_table_sql, language="sql")
                                    #st.code(insert_sql, language="sql")
                        
                                    # ✅ **STEP 3: Align Buttons with Select Boxes**
                                    with col4:
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        if st.button("Create Segment", use_container_width=True, help="Creates a new segment table and inserts selected identifiers"):
                                            try:
                                                session.sql(create_table_sql).collect()  # Create table
                                                session.sql(insert_sql).collect()  # Insert UIDs
                                                st.success(f"✅ Segment `{database}.{schema}.{table_name}` created successfully!")
                                            except Exception as e:
                                                st.error(f"❌ Error executing SQL: {e}")
                        
                                    with col5:
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        if st.button("Append to Segment", use_container_width=True, help="Appends selected identifiers into an existing table"):
                                            try:
                                                session.sql(insert_sql).collect()  # Insert UIDs only
                                                st.success(f"✅ UIDs successfully appended to `{database}.{schema}.{table_name}`!")
                                            except Exception as e:
                                                st.error(f"❌ Error executing SQL: {e}")
                                                
                    elif genre == 'Graph':
                        sigma_graph(res)
            
                    elif genre == 'Sunburst':
                        process_and_generate_sunburst(res, direction="to")

                if st.toggle ("Expl**AI**n Me!",help="Explain journeys and derive insights (summarize paths, describe behaviors and even suggest actions !) by leveraging Cortex AI and the interpretive and generative power of LLMs"):
                                          # Create a temporary view for the aggregated results  
                    prompt = st.text_input("Prompt your question", key="aisqlprompt")

                    # Check if the prompt is valid
                    if prompt and prompt != "None":
                        try:
                            # Step 1: Create a Snowpark DataFrame from res["PATH"]
                            aipath_df = pd.DataFrame(res["PATH"], columns=["PATH"])
                            aipath = session.create_dataframe(aipath_df)
                    
                            # Step 2: Apply the AI aggregation directly
                            from snowflake.snowpark.functions import col, lit, call_function
                    
                            ai_result_df = aipath.select(
                                call_function("AI_AGG", col("PATH"), lit(prompt)).alias("AI_RESPONSE")
                            )
                    
                            # Step 3: Collect and display result
                            explain = ai_result_df.to_pandas()
                    
                            message = st.chat_message("assistant")
                            if not explain.empty:
                                formatted_output = explain.iloc[0, 0]
                                message.markdown(f"**AI Response:**\n\n{formatted_output}")
                            else:
                                message.markdown("No results found for the given prompt.")
                    
                        except Exception as e:
                            st.error(f"An error occurred: {e}")                    
                
                # View Aggregated Paths SQL
                if st.toggle("View SQL for Aggregated Paths"):    
                    st.code(path_to_agg_sql, language='sql')

            if st.toggle("View Detailed Individual Paths"):
                    # Individual Paths SQL
                if unitoftime==None and timeout ==None :
                    
                    path_to_det_sql = f"""
                    select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                        from (select * from {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                        match_recognize(
                        {partitionby} 
                        order by {tmstp} 
                        measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                        all rows per match
                        AFTER MATCH SKIP {overlap}
                        pattern(A{{{minnbbevt},{maxnbbevt}}} B) 
                        define A as true, B AS {evt} IN ({toevt})
                    )  {groupby} """
        
                    # Run the SQL
                    path_to_det_df = session.sql(path_to_det_sql).collect()
        
                    # View Individual Paths Output
                    st.subheader ('Detailed Individual Paths', divider="grey")
                    st.dataframe(path_to_det_df, use_container_width=True)
        
                    # View Individual Paths SQL
                    with st.expander("View SQL for Individual Paths"):    
                        st.code(path_to_det_sql, language='sql')
                        
                elif unitoftime != None and timeout !=None :
                    path_to_det_sql = f"""
                    select {uid}, listagg({display}, ',') within group (order by MSQ) as path
                    from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                    {tmstp}) AS TIMEWINDOW FROM {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                SELECT *FROM sessions) 
                        match_recognize(
                        {partitionby} 
                        order by {tmstp} 
                        measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                        all rows per match
                        AFTER MATCH SKIP {overlap}
                        pattern(A{{{minnbbevt},{maxnbbevt}}} B) 
                        define A as true, B AS {evt} IN ({toevt})
                    )  {groupby} """
                    
                       # Run the SQL
                    path_to_det_df = session.sql(path_to_det_sql).collect()
        
                    # View Individual Paths Output
                    st.subheader ('Detailed Individual Paths', divider="grey")
                    st.dataframe(path_to_det_df, use_container_width=True)
        
                    # View Individual Paths SQL
                    with st.expander("View SQL for Individual Paths"):    
                        st.code(path_to_det_sql, language='sql')
                
            else:
                    st.write("") 
                
        

        # Separate block for PATH FROM 
        elif fromevt.strip("'") != 'Any' and toevt.strip("'")== 'Any':

            path_frm_agg=None
            path_frm_agg_sql=None
            path_frm_det_df=None
            path_frm_det_sql = None
            # Aggregate results for Sankey plot
            if unitoftime==None and timeout ==None :
                
                path_frm_agg_sql = f"""
                select top {topn} path, count(*) as count,array_agg({uid}) as uid_list from (
                    select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                        from (select * from {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                            match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A B{{{minnbbevt},{maxnbbevt}}})
                            define A AS {evt} IN ({fromevt}), B as true)
                    {groupby} ) 
                group by path order by count 
                """
    
                path_frm_agg = session.sql(path_frm_agg_sql).collect()
                
            elif unitoftime != None and timeout !=None :
                 
                path_frm_agg_sql = f"""
                select top {topn} path, count(*) as count,array_agg({uid}) as uid_list from (
                    select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                        from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                    {tmstp}) AS TIMEWINDOW
                 FROM {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                 OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session
                FROM events_with_diff)
                 SELECT *FROM sessions)
                            match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A B{{{minnbbevt},{maxnbbevt}}})
                            define A AS {evt} IN ({fromevt}), B as true)
                    {groupby} ) 
                group by path order by count 
                """
    
                path_frm_agg = session.sql(path_frm_agg_sql).collect()
            
            # If the DataFrame is not empty, show Sankey plot
            res = pd.DataFrame(path_frm_agg)
            import ast
            
            
            if not res.empty:
                def convert_uid_list(uid_entry):
                    if isinstance(uid_entry, str):
                        try:
                            return ast.literal_eval(uid_entry)  # Safely convert string to list
                        except (SyntaxError, ValueError):
                            return []  # Return empty list if conversion fails
                    return uid_entry  # Already a list, return as is
                
                # Apply the function ONLY to UID_LIST while keeping other columns unchanged
                res["UID_LIST"] = res["UID_LIST"].apply(convert_uid_list)
                # Create two columns for layout
                col1, col2,col3 = st.columns([2,7,6])
            
                # Place the toggle in the first column
                with col1:
                    show_details = st.toggle("Show me!", help= "Select a visualization option: Sankey, Tree, Forced Layout Graph or Sunburst.")
                    #show_details = st.toggle("Show me!", help= "Select a visualization option: Sankey, Tree, Forced Layout Graph or Sunburst. In the Tree diagram you can click on a node to select a specific path. Once a path is selected, it will be displayed, showing the sequence of events leading to or from the chosen point. To create a customer segment, expand the CREATE SEGMENT section, choose a database and schema from the available options and enter a segment table name where the user IDs will be stored. After configuring these details, create a new segment by clicking 'Create Segment' which will generate a new table and insert the selected user IDs, or append the selected IDs to an existing table by clicking 'Append to Segment'.")

                # Place the radio button in the second column, but only if the toggle is on
                with col2:
                    if show_details:
                        genre = st.pills(
                            "Choose a visualization:",
                            ["Sankey", "Tree", "Graph", "Sunburst"],
                            label_visibility="collapsed"
                        )
            
                # Place the visualization outside of the columns layout
                if show_details:
                    if genre == 'Tree':
                        target_event = fromevt.strip("'")
                        clicked_node= plot_tree(res, target_event, "from")
                        if clicked_node:
                            selected_path = clicked_node["full_path"]  
                            selected_uids = clicked_node["uids"]  # Directly use cleaned UIDs
                        
                            st.write(f"🔀 Selected path: **{selected_path}**")
                            
                            with st.expander("CREATE SEGMENT"):  # Keep the form visible
                                # **Fetch available databases**
                                sqldb = "SHOW DATABASES"
                                databases = session.sql(sqldb).collect()
                                db0 = pd.DataFrame(databases)
                        
                                # **Row Layout: Database, Schema, Table Name, Create & Append Buttons**
                                col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        
                                # **Database Selection**
                                with col1:
                                    database = st.selectbox("Select Database", key="segmentdb", index=None, 
                                                            placeholder="Choose from list...", options=db0["name"].unique())
                        
                                # **Schema Selection**
                                schema = None
                                if database:
                                    sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                                    schemas = session.sql(sqlschemas).collect()
                                    schema0 = pd.DataFrame(schemas)
                        
                                    with col2:
                                        schema = st.selectbox("Select Schema", key="segmentsch", index=None, 
                                                              placeholder="Choose from list...", options=schema0["name"].unique())
                        
                                # **Table Name Input**
                                table_name = None
                                if database and schema:
                                    with col3:
                                        table_name = st.text_input("Enter Segment Table Name", key="segmenttbl", placeholder="Type table name...")
                        
                                # **Ensure all fields are filled before proceeding**
                                if database and schema and table_name and selected_uids:
                                    # ✅ **STEP 1: GENERATE SQL STATEMENTS**
                                    create_table_sql = f"""
                                    CREATE OR REPLACE TABLE {database}.{schema}.{table_name} (
                                        ID STRING
                                    )
                                    """
                        
                                    values = ", ".join([f"('{uid}')" for uid in selected_uids])  # Format UIDs
                                    insert_sql = f"INSERT INTO {database}.{schema}.{table_name} (ID) VALUES {values};"
                        
                                    # ✅ **STEP 2: SHOW SQL PREVIEW**
                                    #st.write("#### SQL Preview:")
                                    #st.code(create_table_sql, language="sql")
                                    #st.code(insert_sql, language="sql")
                        
                                    # ✅ **STEP 3: Align Buttons with Select Boxes**
                                    with col4:
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        if st.button("Create Segment", use_container_width=True, help="Creates a new segment table and inserts selected identifiers"):
                                            try:
                                                session.sql(create_table_sql).collect()  # Create table
                                                session.sql(insert_sql).collect()  # Insert UIDs
                                                st.success(f"✅ Segment `{database}.{schema}.{table_name}` created successfully!")
                                            except Exception as e:
                                                st.error(f"❌ Error executing SQL: {e}")
                        
                                    with col5:
                                        st.markdown("<br>", unsafe_allow_html=True)
                                        if st.button("Append to Segment", use_container_width=True, help="Appends selected identifiers into an existing table"):
                                            try:
                                                session.sql(insert_sql).collect()  # Insert UIDs only
                                                st.success(f"✅ UIDs successfully appended to `{database}.{schema}.{table_name}`!")
                                            except Exception as e:
                                                st.error(f"❌ Error executing SQL: {e}")
                    
                    elif genre == 'Sankey':
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            percentage = st.slider("Display Top % of Paths", 1, 100, 100)
                        with col2:
                            st.write("")                        
                         # 🔹 **STEP 2: Reset Session State on Query Change**
                        current_df_hash = hash(res.to_json())
                        if "last_df_hash" not in st.session_state or st.session_state["last_df_hash"] != current_df_hash:
                            st.session_state["last_df_hash"] = current_df_hash  # Update hash
                            # Reset state variables
                            st.session_state["clicked_sankey"] = None
                            st.session_state["clicked_source"] = None
                            st.session_state["clicked_target"] = None
                            st.session_state["selected_uids"] = set()
                            st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                            st.session_state["sankey_chart"] = None
                            st.session_state["sankey_links"] = {}
                            st.session_state["sankey_labels"] = []
                            st.session_state["sortedEventList"] = []
                            
                            #st.info("ℹ️ **Query executed! Resetting selections & memory.**")
                        # Session state to store extracted user data
                        if "selected_paths_df" not in st.session_state:
                            st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                        
                        clicked_sankey = sankey_chart(res, direction="from", topN_percentage=percentage)
                        
                        if clicked_sankey:
                            sankeyLabel = st.session_state.get("sankey_labels", [])
                            sortedEventList = st.session_state.get("sortedEventList", [])
                            sankeyLinks = st.session_state.get("sankey_links", {})
                            if "source" in clicked_sankey and "target" in clicked_sankey:
                                source_index = clicked_sankey["source"]
                                target_index = clicked_sankey["target"]
                                clicked_source = sortedEventList[source_index]
                                clicked_target = sortedEventList[target_index]
                                st.caption(f"Selected Edge: {clicked_source.split('_', 1)[1]} → {clicked_target.split('_', 1)[1]}")
                                valuePair = f"{clicked_source}+{clicked_target}"
                                extracted_uids = sankeyLinks.get(valuePair, {}).get("uids", [])
                                #st.write(f"👤 Extracted UIDs: {extracted_uids}")
                                flattened_uids = set()
                                for uid_list in extracted_uids:
                                    if isinstance(uid_list, list):
                                        flattened_uids.update(map(str, uid_list))
                                    else:
                                        flattened_uids.add(str(uid_list))
                                # Store in session state (initialize if not present)
                                if "selected_uids" not in st.session_state:
                                    st.session_state["selected_uids"] = flattened_uids
                                else:
                                    st.session_state["selected_uids"].update(flattened_uids)  # Accumulate UIDs
                                # Update DataFrame for UI display
                                new_entry = pd.DataFrame([{"Source": clicked_source, "Target": clicked_target, "User_IDs": list(flattened_uids)}])
                                if "selected_paths_df" not in st.session_state:
                                    st.session_state["selected_paths_df"] = new_entry
                                else:
                                    st.session_state["selected_paths_df"] = pd.concat(
                                        [st.session_state["selected_paths_df"], new_entry], ignore_index=True
                                    )
                                #st.write(f"👤 Extracted UIDs: {list(flattened_uids)}")
                        #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        # 🔹 : Add a Manual Reset Button**
                        # if st.button("🔄 Reset Selection", use_container_width=False):
                        #     # Reset all relevant session state variables
                        #     st.session_state["selected_uids"] = set()
                        #     st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                        #     st.session_state["clicked_sankey"] = None
                        #     st.session_state["clicked_source"] = None
                        #     st.session_state["clicked_target"] = None
                        #     st.session_state["sankey_links"] = {}
                        #     st.session_state["sankey_labels"] = []
                        #     st.session_state["sortedEventList"] = []
                            
                        #     st.session_state["refresh_key"] = not st.session_state.get("refresh_key", False)
                        #     #st.success("✅ Selections & memory cleared!")
                        #     #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        
                        # with st.expander("CREATE SEGMENT"):
                        #     # Fetch available databases
                        #     sqldb = "SHOW DATABASES"
                        #     databases = session.sql(sqldb).collect()
                        #     db0 = pd.DataFrame(databases)
                        #     col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        #     # Database Selection
                        #     with col1:
                        #         database = st.selectbox("Select Database", key="segmentdb", index=None, 
                        #                                 placeholder="Choose from list...", options=db0["name"].unique())
                        #     # Schema Selection
                        #     schema = None
                        #     if database:
                        #         sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                        #         schemas = session.sql(sqlschemas).collect()
                        #         schema0 = pd.DataFrame(schemas)
                        #         with col2:
                        #             schema = st.selectbox("Select Schema", key="segmentsch", index=None, 
                        #                                 placeholder="Choose from list...", options=schema0["name"].unique())
                        #     # Table Name Input
                        #     table_name = None
                        #     if database and schema:
                        #         with col3:
                        #             table_name = st.text_input("Enter Segment Table Name", key="segmenttbl", placeholder="Type table name...")
                        #     # ✅ Use selected UIDs directly from session_state
                        #     selected_uids = st.session_state.get("selected_uids", set())
                        #     if database and schema and table_name and selected_uids:
                        #         #st.write(f"🔍 Extracted UIDs for insertion: {selected_uids}")
                        #         create_table_sql = f"""
                        #         CREATE OR REPLACE TABLE {database}.{schema}.{table_name} (
                        #             ID STRING
                        #         )
                        #         """
                        #         # Convert UIDs to SQL-safe format
                        #         values = ", ".join([f"('{uid}')" for uid in selected_uids])
                        #         insert_sql = f"INSERT INTO {database}.{schema}.{table_name} (ID) VALUES {values};" if values else ""
                        #         #st.write(f"🔍 SQL Insert Preview:\n{insert_sql}")  # Debugging
                        #         with col4:
                        #             if st.button("Create Segment", use_container_width=True):
                        #                 try:
                        #                     session.sql(create_table_sql).collect()
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                     st.success(f"✅ Segment `{database}.{schema}.{table_name}` created successfully!")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
                        #         with col5:
                        #             if st.button("Append to Segment", use_container_width=True):
                        #                 try:
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                         st.success(f"✅ UIDs successfully appended to `{database}.{schema}.{table_name}`!")
                        #                     else:
                        #                         st.warning("⚠️ No UIDs selected to append.")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
                    elif genre == 'Graph':
                        sigma_graph(res)
            
                    elif genre == 'Sunburst':
                        process_and_generate_sunburst(res, direction="from")
                        
            if st.toggle ("Expl**AI**n Me!",help="Explain journeys and derive insights (summarize paths, describe behaviors and even suggest actions !) by leveraging Cortex AI and the interpretive and generative power of LLMs"):
                                          # Create a temporary view for the aggregated results  
                    prompt = st.text_input("Prompt your question", key="aisqlprompt")

                    # Check if the prompt is valid
                    if prompt and prompt != "None":
                        try:
                            # Step 1: Create a Snowpark DataFrame from res["PATH"]
                            aipath_df = pd.DataFrame(res["PATH"], columns=["PATH"])
                            aipath = session.create_dataframe(aipath_df)
                    
                            # Step 2: Apply the AI aggregation directly
                            from snowflake.snowpark.functions import col, lit, call_function
                    
                            ai_result_df = aipath.select(
                                call_function("AI_AGG", col("PATH"), lit(prompt)).alias("AI_RESPONSE")
                            )
                    
                            # Step 3: Collect and display result
                            explain = ai_result_df.to_pandas()
                    
                            message = st.chat_message("assistant")
                            if not explain.empty:
                                formatted_output = explain.iloc[0, 0]
                                message.markdown(f"**AI Response:**\n\n{formatted_output}")
                            else:
                                message.markdown("No results found for the given prompt.")
                    
                        except Exception as e:
                            st.error(f"An error occurred: {e}")  
                            
                # View Aggregated Paths SQL
            if st.toggle("View SQL for Aggregated Paths"):     
                    st.code(path_frm_agg_sql, language='sql')

            if st.toggle("View Detailed Individual Paths"):
        # Individual Paths SQL
                if unitoftime==None and timeout ==None :
                    path_frm_det_sql = f"""
                    select {uid}, listagg({display}, ',') within group (order by MSQ) as path
                    from  (select * from {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                        match_recognize(
                        {partitionby} 
                        order by {tmstp} 
                        measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                        all rows per match
                        AFTER MATCH SKIP {overlap}
                        pattern(A B{{{minnbbevt},{maxnbbevt}}}) 
                        define B as true, A AS {evt} IN ({fromevt})
                    )  {groupby} """
                    
                    # Run the SQL
                    path_frm_det_df = session.sql(path_frm_det_sql).collect()
                    
                    # View Individual Paths Output    
                    st.subheader ('Detailed Individual Paths', divider="grey")
                    st.dataframe(path_frm_det_df, use_container_width=True)
        
                    # View Individual Paths SQL
                    with st.expander("View SQL for Individual Paths"):    
                        st.code(path_frm_det_sql, language='sql')
                elif unitoftime != None and timeout !=None :
                    path_frm_det_sql = f"""
                    select {uid}, listagg({display}, ',') within group (order by MSQ) as path
                    from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                    {tmstp}) AS TIMEWINDOW FROM {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                SELECT *FROM sessions) 
                        match_recognize(
                        {partitionby} 
                        order by {tmstp} 
                        measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                        all rows per match
                        AFTER MATCH SKIP {overlap}
                        pattern(A B{{{minnbbevt},{maxnbbevt}}}) 
                        define B as true, A AS {evt} IN ({fromevt})
                    )  {groupby} """
                    
                    # Run the SQL
                    path_frm_det_df = session.sql(path_frm_det_sql).collect()
                    
                    # View Individual Paths Output    
                    st.subheader ('', divider="grey")
                    st.dataframe(path_frm_det_df, use_container_width=True)
        
                    # View Individual Paths SQL
                    with st.expander("View SQL for Individual Paths"):    
                        st.code(path_frm_det_sql, language='sql')
                
            else:
                st.write("")
                
        # Separate block for PATH BETWEEN 
        elif fromevt.strip("'") != 'Any' and toevt.strip("'") != 'Any':
            
            path_betw_agg=None
            path_betw_agg_sql=None
            path_betw_det_df=None
            path_betw_det_sql = None
            # Aggregate results for Sankey plot
            if unitoftime==None and timeout ==None :
                
                path_betw_agg_sql = f"""
                select top {topn} path, count(*) as count, array_agg({uid}) as uid_list from (
                    select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                        from (select * from {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                            match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A X{{{minnbbevt},{maxnbbevt}}} B) 
                            define  A AS {evt} IN ({fromevt}), X as true, B AS {evt} IN ({toevt}))
                    {groupby} ) 
                group by path order by count 
                """
              
                path_betw_agg = session.sql(path_betw_agg_sql).collect()
                
            elif unitoftime != None and timeout !=None :
                 
                path_betw_agg_sql = f"""
                select top {topn} path, count(*) as count, array_agg({uid}) as uid_list from (
                    select {uid},  listagg({display}, ', ') within group (order by MSQ)  as path
                        from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                    {tmstp}) AS TIMEWINDOW
                 FROM {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                 OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session
                FROM events_with_diff)
                 SELECT *FROM sessions)
                            match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A X{{{minnbbevt},{maxnbbevt}}} B) 
                            define  A AS {evt} IN ({fromevt}), X as true, B AS {evt} IN ({toevt}))
                    {groupby} ) 
                group by path order by count 
                """
               
                path_betw_agg = session.sql(path_betw_agg_sql).collect()
            
            # If the DataFrame is not empty, show Sankey plot
            res = pd.DataFrame(path_betw_agg)
            import ast
           
            if not res.empty:
                def convert_uid_list(uid_entry):
                    if isinstance(uid_entry, str):
                        try:
                            return ast.literal_eval(uid_entry)  # Safely convert string to list
                        except (SyntaxError, ValueError):
                            return []  # Return empty list if conversion fails
                    return uid_entry  # Already a list, return as is
                
                # Apply the function ONLY to UID_LIST while keeping other columns unchanged
                res["UID_LIST"] = res["UID_LIST"].apply(convert_uid_list)
                # Create two columns for layout
                col1, col2 = st.columns([1, 3])
            
                # Place the toggle in the first column
                with col1:
                    show_details = st.toggle("Show me!", help="Select a visualization option: Sankey, Forced Layout Graph or Sunburst")
            
                # Place the radio button in the second column, but only if the toggle is on
                with col2:
                    if show_details:
                        genre = st.pills("Choose a visualization:",["Sankey", "Graph", "Sunburst"],label_visibility="collapsed" )
            
                # Place the visualization outside of the columns layout
                if show_details:
            
                    if genre == 'Graph':
                        sigma_graph(res)
                    
                    elif genre == 'Sankey':

                         # 🔹 **STEP 2: Reset Session State on Query Change**
                        current_df_hash = hash(res.to_json())
                        if "last_df_hash" not in st.session_state or st.session_state["last_df_hash"] != current_df_hash:
                            st.session_state["last_df_hash"] = current_df_hash  # Update hash
                            # Reset state variables
                            st.session_state["clicked_sankey"] = None
                            st.session_state["clicked_source"] = None
                            st.session_state["clicked_target"] = None
                            st.session_state["selected_uids"] = set()
                            st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                            st.session_state["sankey_chart"] = None
                            st.session_state["sankey_links"] = {}
                            st.session_state["sankey_labels"] = []
                            st.session_state["sortedEventList"] = []
                            
                            #st.info("ℹ️ **Query executed! Resetting selections & memory.**")
                        # Session state to store extracted user data
                        if "selected_paths_df" not in st.session_state:
                            st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                        
                        clicked_sankey = sankey_chart(res, direction="to")
                        
                        if clicked_sankey:
                            sankeyLabel = st.session_state.get("sankey_labels", [])
                            sortedEventList = st.session_state.get("sortedEventList", [])
                            sankeyLinks = st.session_state.get("sankey_links", {})
                            if "source" in clicked_sankey and "target" in clicked_sankey:
                                source_index = clicked_sankey["source"]
                                target_index = clicked_sankey["target"]
                                clicked_source = sortedEventList[source_index]
                                clicked_target = sortedEventList[target_index]
                                st.caption(f"Selected Edge: {clicked_source.split('_', 1)[1]} → {clicked_target.split('_', 1)[1]}")
                                valuePair = f"{clicked_source}+{clicked_target}"
                                extracted_uids = sankeyLinks.get(valuePair, {}).get("uids", [])
                                #st.write(f"👤 Extracted UIDs: {extracted_uids}")
                                flattened_uids = set()
                                for uid_list in extracted_uids:
                                    if isinstance(uid_list, list):
                                        flattened_uids.update(map(str, uid_list))
                                    else:
                                        flattened_uids.add(str(uid_list))
                                # Store in session state (initialize if not present)
                                if "selected_uids" not in st.session_state:
                                    st.session_state["selected_uids"] = flattened_uids
                                else:
                                    st.session_state["selected_uids"].update(flattened_uids)  # Accumulate UIDs
                                # Update DataFrame for UI display
                                new_entry = pd.DataFrame([{"Source": clicked_source, "Target": clicked_target, "User_IDs": list(flattened_uids)}])
                                if "selected_paths_df" not in st.session_state:
                                    st.session_state["selected_paths_df"] = new_entry
                                else:
                                    st.session_state["selected_paths_df"] = pd.concat(
                                        [st.session_state["selected_paths_df"], new_entry], ignore_index=True
                                    )
                                #st.write(f"👤 Extracted UIDs: {list(flattened_uids)}")
                        #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        # 🔹 : Add a Manual Reset Button**
                        # if st.button("🔄 Reset Selection", use_container_width=False):
                        #     # Reset all relevant session state variables
                        #     st.session_state["selected_uids"] = set()
                        #     st.session_state["selected_paths_df"] = pd.DataFrame(columns=["Source", "Target", "User_IDs"])
                        #     st.session_state["clicked_sankey"] = None
                        #     st.session_state["clicked_source"] = None
                        #     st.session_state["clicked_target"] = None
                        #     st.session_state["sankey_links"] = {}
                        #     st.session_state["sankey_labels"] = []
                        #     st.session_state["sortedEventList"] = []
                            
                        #     st.session_state["refresh_key"] = not st.session_state.get("refresh_key", False)
                        #     #st.success("✅ Selections & memory cleared!")
                        #     #st.write(f"📌 Current Distinct UIDs in Memory: {st.session_state['selected_uids']}")
                        
                        # with st.expander("CREATE SEGMENT"):
                        #     # Fetch available databases
                        #     sqldb = "SHOW DATABASES"
                        #     databases = session.sql(sqldb).collect()
                        #     db0 = pd.DataFrame(databases)
                        #     col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        #     # Database Selection
                        #     with col1:
                        #         database = st.selectbox("Select Database", key="segmentdb", index=None, 
                        #                                 placeholder="Choose from list...", options=db0["name"].unique())
                        #     # Schema Selection
                        #     schema = None
                        #     if database:
                        #         sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                        #         schemas = session.sql(sqlschemas).collect()
                        #         schema0 = pd.DataFrame(schemas)
                        #         with col2:
                        #             schema = st.selectbox("Select Schema", key="segmentsch", index=None, 
                        #                                 placeholder="Choose from list...", options=schema0["name"].unique())
                        #     # Table Name Input
                        #     table_name = None
                        #     if database and schema:
                        #         with col3:
                        #             table_name = st.text_input("Enter Segment Table Name", key="segmenttbl", placeholder="Type table name...")
                        #     # ✅ Use selected UIDs directly from session_state
                        #     selected_uids = st.session_state.get("selected_uids", set())
                        #     if database and schema and table_name and selected_uids:
                        #         #st.write(f"🔍 Extracted UIDs for insertion: {selected_uids}")
                        #         create_table_sql = f"""
                        #         CREATE OR REPLACE TABLE {database}.{schema}.{table_name} (
                        #             ID STRING
                        #         )
                        #         """
                        #         # Convert UIDs to SQL-safe format
                        #         values = ", ".join([f"('{uid}')" for uid in selected_uids])
                        #         insert_sql = f"INSERT INTO {database}.{schema}.{table_name} (ID) VALUES {values};" if values else ""
                        #         #st.write(f"🔍 SQL Insert Preview:\n{insert_sql}")  # Debugging
                        #         with col4:
                        #             if st.button("Create Segment", use_container_width=True):
                        #                 try:
                        #                     session.sql(create_table_sql).collect()
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                     st.success(f"✅ Segment `{database}.{schema}.{table_name}` created successfully!")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
                        #         with col5:
                        #             if st.button("Append to Segment", use_container_width=True):
                        #                 try:
                        #                     if values:
                        #                         session.sql(insert_sql).collect()
                        #                         st.success(f"✅ IDs successfully appended to `{database}.{schema}.{table_name}`!")
                        #                     else:
                        #                         st.warning("⚠️ No IDs selected to append.")
                        #                 except Exception as e:
                        #                     st.error(f"❌ Error executing SQL: {e}")
            
                    elif genre == 'Sunburst':
                        process_and_generate_sunburst(res, direction="to")
                        
            if st.toggle ("Expl**AI**n Me!",help="Explain journeys and derive insights (summarize paths, describe behaviors and even suggest actions !) by leveraging Cortex AI and the interpretive and generative power of LLMs"):
                                          # Create a temporary view for the aggregated results  
                    prompt = st.text_input("Prompt your question", key="aisqlprompt")

                    # Check if the prompt is valid
                    if prompt and prompt != "None":
                        try:
                            # Step 1: Create a Snowpark DataFrame from res["PATH"]
                            aipath_df = pd.DataFrame(res["PATH"], columns=["PATH"])
                            aipath = session.create_dataframe(aipath_df)
                    
                            # Step 2: Apply the AI aggregation directly
                            from snowflake.snowpark.functions import col, lit, call_function
                    
                            ai_result_df = aipath.select(
                                call_function("AI_AGG", col("PATH"), lit(prompt)).alias("AI_RESPONSE")
                            )
                    
                            # Step 3: Collect and display result
                            explain = ai_result_df.to_pandas()
                    
                            message = st.chat_message("assistant")
                            if not explain.empty:
                                formatted_output = explain.iloc[0, 0]
                                message.markdown(f"**AI Response:**\n\n{formatted_output}")
                            else:
                                message.markdown("No results found for the given prompt.")
                    
                        except Exception as e:
                            st.error(f"An error occurred: {e}")                  
                # View Aggregated Paths SQL
            if st.toggle("View SQL for Aggregated Paths"):      
                st.code(path_betw_agg_sql, language='sql')

            if st.toggle("View Detailed Individual Paths"):
        # Individual Paths SQL
                if unitoftime==None and timeout ==None :
                    path_betw_det_sql = f"""
                    select {uid}, listagg({display}, ',') within group (order by MSQ) as path
                    from  (select * from {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                        match_recognize(
                        {partitionby} 
                        order by {tmstp} 
                        measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                        all rows per match
                        AFTER MATCH SKIP {overlap}
                        pattern(A X{{{minnbbevt},{maxnbbevt}}} B) 
                        define  A AS {evt} IN ({fromevt}), X as true, B AS {evt} IN ({toevt})
                    )  {groupby} """
                    
                    # Run the SQL
                    path_betw_det_df = session.sql(path_betw_det_sql).collect()
                    
                    # View Individual Paths Output    
                    st.subheader ('Detailed Individual Paths', divider="grey")
                    st.dataframe(path_betw_det_df, use_container_width=True)
        
                    # View Individual Paths SQL
                    with st.expander("View SQL for Individual Paths"):    
                        st.code(path_betw_det_sql, language='sql')
                elif unitoftime != None and timeout !=None :
                    path_betw_det_sql = f"""
                    select {uid}, listagg({display}, ',') within group (order by MSQ) as path
                    from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                    {tmstp}) AS TIMEWINDOW FROM {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                SELECT *FROM sessions) 
                        match_recognize(
                        {partitionby} 
                        order by {tmstp} 
                        measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                        all rows per match
                        AFTER MATCH SKIP {overlap}
                        pattern(A X{{{minnbbevt},{maxnbbevt}}} B) 
                        define  A AS {evt} IN ({fromevt}), X as true, B AS {evt} IN ({toevt})
                    )  {groupby} """
                    
                    # Run the SQL
                    path_betw_det_df = session.sql(path_betw_det_sql).collect()
                    
                    # View Individual Paths Output    
                    st.subheader ('Detailed Individual Paths', divider="grey")
                    st.dataframe(path_betw_det_df, use_container_width=True)
        
                    # View Individual Paths SQL
                    with st.expander("View SQL for Individual Paths"):    
                        st.code(path_betw_det_sql, language='sql')
                
            else:
                st.write("")
            

        elif fromevt.strip("'") == 'Any' and toevt.strip("'") == 'Any':
            st.warning("This is tuple generator")
            
            path_tupl_agg=None
            path_tupl_agg_sql=None
            
            # Aggregate results for Sankey plot
            if unitoftime==None and timeout ==None :
                path_tupl_agg_sql = f"""
                select top {topn} path, count(*) as count from (
                    select {uid},  listagg({display}, ', ') within group (order by MSQ) as path
                        from (select * from {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                            match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A{{{minnbbevt},{maxnbbevt}}}) 
                            define  A as true)
                    {groupby} ) 
                group by path order by count 
                """
                path_tupl_agg = session.sql(path_tupl_agg_sql).collect()
                
            elif unitoftime != None and timeout !=None :
                path_tupl_agg_sql = f"""
                select top {topn} path, count(*) as count from (
                    select {uid},  listagg({display}, ', ') within group (order by MSQ) as path
                        from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                    {tmstp}) AS TIMEWINDOW FROM {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                SELECT *FROM sessions) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp}  
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap} 
                            pattern(A{{{minnbbevt},{maxnbbevt}}}) 
                            define  A as true)
                    {groupby} ) 
                group by path order by count 
                """
                path_tupl_agg = session.sql(path_tupl_agg_sql).collect()
                    # If the DataFrame is not empty, show Sankey plot
            res = pd.DataFrame(path_tupl_agg)
            if not res.empty:
                # Create two columns for layout
                col1, col2 = st.columns([1, 3])
            
                # Place the toggle in the first column
                with col1:
                    show_details = st.toggle("Show me!", help="Select a visualization option: Sankey, Forced Layout Graph or Sunburst")
            
                # Place the radio button in the second column, but only if the toggle is on
                with col2:
                    if show_details:
                        genre = st.radio(
                            "Choose a visualization:",
                            ["Sankey", "Graph", "Sunburst"],
                            index=None,
                            horizontal=True,
                            label_visibility="collapsed"
                        )
            
                # Place the visualization outside of the columns layout
                if show_details:
                    if genre == 'Sankey':
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            sli = st.slider("Path Count Filter", 0, topn, topn)
                        with col2:
                            st.write("")
                        sankeyPlot(res, "to", "", sli)
            
                    elif genre == 'Graph':
                        sigma_graph(res)
            
                    elif genre == 'Sunburst':
                        process_and_generate_sunburst(res, direction="from")
     
                
        else:
            st.write("Please select appropriate options for 'from' and 'to'")
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
        
#--------------------------------------
#COMPARE TAB
#--------------------------------------
# PATH COMPARISON
# DEFINE ONE OR TWO TARGET SETS USING THE CONTROL PANE
# 2 MODES AVAILABLE
# COMPLEMENT : Define one reference target set and compare importance of events from events from everything but this target set
# UNION : Define two target sets and compare importance of events for each set

with tab2:
    with st.expander("Comparison Mode"):
        st.caption("**Complement Mode**(default mode): This mode focuses on ensemble-based analysis by comparing a population defined by a (**Reference**) set of paths against everything outside of it.")
        st.caption("**Union Mode**: This mode performs a comparative analysis between two defined (**Reference**) and (**Compared**) sets of paths (i.e., populations), accounting for any potential overlap.")
        mode=st.radio(
    "Mode:",
    ["Complement", "Union"],
    key="horizontal", horizontal=True,
           label_visibility="collapsed"
)
        
    with st.expander("Input Parameters (Reference)"):
            
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
                database = st.selectbox('Select Database', key='comparerefdb', index=None, 
                                        placeholder="Choose from list...", options=db0['name'].unique())
            
            # **Schema Selection (Only if a database is selected)**
            if database:
                sqlschemas = f"SHOW SCHEMAS IN DATABASE {database}"
                schemas = session.sql(sqlschemas).collect()
                schema0 = pd.DataFrame(schemas)
            
                with col2:
                    schema = st.selectbox('Select Schema', key='comparerefsch', index=None, 
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
                    tbl = st.selectbox('Select Event Table or View', key='comparereftbl', index=None, 
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

            col1, col2, col3 = st.columns([4,4,4])
            with col1:
                uid = st.selectbox('Select identifier column', colsdf, index=None,  key='uidcompareref',placeholder="Choose from list...", help="The identifier column is the unique identifier for each event record. This may be a user id, a customer id, a process id, etc ...it is used to partition the input set of rows before pattern matching.")
            with col2: 
                evt = st.selectbox('Select event column', colsdf, index=None, key='evtcompareref', placeholder="Choose from list...",help="The event column contains the actual events that will be used in the path analysis.")
            with col3:
                tmstp = st.selectbox('Select timestamp column', colsdf, index=None, key='tsmtpcompareref',placeholder="Choose from list...",help="The timestamp column contains the timestamp associated to the event. This is the value that will be used for sequentially ordering the dataset.")
            
            #Get Distinct Events Of Interest from Event Table
            if (uid != None and evt != None and tmstp != None):
            # Get Distinct Events Of Interest from Event Table
                EOI = f"SELECT DISTINCT {evt} FROM {tbl} ORDER BY {evt}"
                        # Get start EOI :
                start = session.sql(EOI).collect()
            # Get end EOI :
                end = session.sql(EOI).collect()
            # Get excluded EOI :
                excl =session.sql(EOI).collect()
        
                # Write query output in a pandas dataframe
                startdf0 = pd.DataFrame(start)
                enddf0 = pd.DataFrame(end)
                excl0 =pd.DataFrame(excl)   


            #Add "any" to the distinct events list
                any_row = {evt: 'Any'}

        # Convert the any_row to a DataFrame and append it using pd.concat
                any_row_df = pd.DataFrame([any_row])
        # Add any to list of FROM 
                startdf1 = pd.concat([ any_row_df, startdf0], ignore_index=True)
        # Add any to list of TO
                enddf1 = pd.concat([ any_row_df, enddf0], ignore_index=True)

        #EVENTS PATTERN
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Events Pattern</h2>
        <hr style='margin-top: -8px;margin-bottom: 5px;'>
        """, unsafe_allow_html=True)
                                
                # Add a None placeholder to force user to select an event
                options_with_placeholder_from = ["🔍"] + startdf1[evt].unique().tolist()
                options_with_placeholder_to = ["🔎"] + enddf1[evt].unique().tolist()
                    
                col1, col2, col3,col4 = st.columns([4,4,2,2])

                with col1:
                    frm = st.multiselect('Select events FROM:', options=options_with_placeholder_from[1:], key='evtfrmref',default=[], help="Select one or more events of interest to visualize paths FROM the selected point(s). 'Any' matches all values.")
                    #filtered_frmevt = startdf1[(startdf1[evt] == frm)]
                    #fromevt = filtered_frmevt.iloc[0, 0]
                    if frm != "🔍":
                        fromevt= ", ".join([f"'{value}'" for value in frm])
 
                    else:
                        fromevt = None  # Set to None if the placeholder is still selected
                    
                    
                with col2:
                    to = st.multiselect('Select events TO:', options=options_with_placeholder_to[1:],  key='evttoref',default=[],help="Select one or more events of interest to visualize paths TO the selected point(s). 'Any' matches all values.")
                    #filtered_toevt = enddf1[(enddf1[evt] == to)]
                    #toevt =filtered_toevt.iloc[0, 0]
                    if to != "🔎":
                        toevt = ", ".join([f"'{value}'" for value in to])
    
                    else:
                        toevt = None  # Set to None if the placeholder is still selected

                with col3:
                    minnbbevt = st.number_input("Min # events", value=0,  key='minnbbevtref',placeholder="Type a number...",help="Select the minimum number of events either preceding or following the event(s) of interest.")

                with col4:
                    maxnbbevt = st.number_input("Max # events", value=5, min_value=1, key='maxnbbevtref' ,placeholder="Type a number...",help="Select the maximum number of events either preceding or following the event(s) of interest.")
                
                col1, col2 = st.columns([5,10])
                with col1:
                    checkoverlap = st.toggle("Allow overlap", key="checkoverlapcompare1",help="Specifies the pattern-matching mode. Toggled-on allows OVERLAP and finds every occurrence of pattern in partition, regardless of whether it is part of a previously found match. One row can match multiple symbols in a given matched pattern.Toggled-off does not allow OVERLAP and starts next pattern search at row that follows last pattern match.")
                    overlap = 'PAST LAST ROW'
                    if checkoverlap:
                        overlap='TO NEXT ROW'
                with col2:
                    st.write("")


                #--------------------------------------
                #DATE RANGE
                #--------------------------------------
                with st.container():
                    st.markdown("""
        <h2 style='font-size: 14px; margin-bottom: 0px;'>Date range</h2>
        <hr style='margin-top: -8px;margin-bottom: 5px;'>
        """, unsafe_allow_html=True)

                col1, col2, col3 = st.columns([2.4,2.4,10])

                # SQL query to get the min start date
                minstartdt = f"SELECT   TO_VARCHAR(MIN ({tmstp}), 'YYYY/MM/DD') FROM {tbl}"
                # Get min start date :
                defstartdt = session.sql(minstartdt).collect()
                defstartdt_str = defstartdt[0][0] 
                defstartdt1 = datetime.datetime.strptime(defstartdt_str, '%Y/%m/%d').date()
                
                with col1:
                    startdt_input = st.date_input('Start date', key='startdtref', value=defstartdt1)
                with col2:
                    enddt_input = st.date_input('End date', key='enddtref',help="Apply a time window to the data set to find paths on a specific date range or over the entire lifespan of your data (default values)")
                with col3: 
                    st.write("")
               
                #--------------------------------------
                #SESSION
                #--------------------------------------
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Session (optional)</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True,help= "A session, in the context of customer journey analysis, refers to a defined period of activity during which a customer interacts or events occur. It includes all events, actions, or transactions performed by the user within this timeframe. If a session field is present in the event table, select it from the 'Column' tab below. If no session field is available, use the 'Sessionize' tab to create unique session identifiers by grouping events based on time gaps (e.g., a gap of more than 30 minutes starts a new session). Once a session is selected or created, it will be used alongside the unique identifier to partition the input rows before applying pattern matching.")
                tab11, tab22 = st.tabs(["Column", "Sessionize"])
               
                with tab11:
                    col1, col2  = st.columns([5,10])
                    with col1:
                            sess = st.selectbox('Select session column', colsdf, index=None, key='sessref', placeholder="Choose from list...",help="If a session field is available within the event table.")
                    
                    with col2:
                        st.write("")
                
                with tab22:
                    col1, col2, col3 = st.columns([2.4,2.4,10])
                    with col1:
                        unitoftime =st.selectbox( "Unit of time",
                    ("SECOND", "MINUTE", "HOUR","DAY"),index=None, key='unitoftimeref', placeholder="Choose from list", help="Select the unit of time of the session time window.")
                  
                    with col2:
                        timeout=  st.number_input( "Insert a timeout value",key='timeoutref',value=None, min_value=1, format="%d", placeholder="Type a number",help="Value of the session time window.")
                if sess == None and unitoftime==None and timeout==None: 
                        partitionby = f"partition by {uid}"
                        groupby = f"group by {uid}, match_number "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp])]['COLUMN_NAME']
                
                elif sess != None and unitoftime==None and timeout==None:
                        partitionby=f"partition by {uid},{sess} "
                        groupby = f"group by {uid}, match_number,{sess} "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp,sess])]['COLUMN_NAME']
    
                elif sess == None and unitoftime !=None and timeout !=None:
                        partitionby=f"partition by {uid},SESSION "
                        groupby = f"group by {uid}, match_number,SESSION "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf[~colsdf['COLUMN_NAME'].isin([uid, evt, tmstp])]['COLUMN_NAME']
                else :
                    st.write("")
                #--------------------------------------
                #FILTERS
                #--------------------------------------

                #initialize sql_where_clause
                sql_where_clause = ""

                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Filters</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)

                    col1, col2  =st.columns([5,10])
                    with col1:
                # Exclude Events
                        excl1 = st.multiselect('Exclude event(s) - optional', excl0,placeholder="Select event(s)...", key='reference',help="Event(s) to be excluded from the pattern evaluation and the ouput.") 

                        if not excl1:
                          excl3 = "''"
                        else:
                         excl3= ', '.join([f"'{excl2}'" for excl2 in excl1])
                    
                                            
                                    # Ensure we have remaining columns before allowing additional filters
                if not remaining_columns.empty:
                    col1, col2 = st.columns([5, 10])
                
                    with col1:
                        checkfilters = st.toggle("Additional filters", key='filterscompareref',help="Apply one or many conditional filters to the input data used in the path and pattern analysis.")
                    with col2:
                        st.write("")    
                else:
                    checkfilters = False  # Disable filters when no columns are left
                
                # Only execute filter logic if there are available columns AND the toggle is enabled
                if checkfilters and not remaining_columns.empty:
                
                    with st.container():
                        # Helper function to fetch distinct values from a column
                        def fetch_distinct_values(column):
                            """Query the distinct values for a column, except for dates"""
                            query = f"SELECT DISTINCT {column} FROM {tbl}"
                            result = session.sql(query).collect()
                            distinct_values = [row[column] for row in result]
                            return distinct_values
                
                        # Helper function to display operator selection based on column data type
                        def get_operator_input(col_name, col_data_type, filter_index):
                            """ Returns the operator for filtering based on column type """
                            operator_key = f"{col_name}_operator_{filter_index}"  # Ensure unique key
                
                            if col_data_type in ['NUMBER', 'FLOAT', 'INT']:
                                operator = st.selectbox("Operator", ['=', '<', '<=', '>', '>=', '!=', 'IN'], key=operator_key)
                            elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                                operator = st.selectbox("Operator", ['<=', '>=', '='], key=operator_key)
                            else:  # For string or categorical columns
                                operator = st.selectbox("Operator", ['=', '!=', 'IN'], key=operator_key)
                            return operator
                
                        # Helper function to display value input based on column data type
                        def get_value_input(col_name, col_data_type, operator, filter_index):
                            """ Returns the value for filtering based on column type """
                            value_key = f"{col_name}_value_{filter_index}"  # Ensure unique key
                
                            if operator == 'IN':
                                distinct_values = fetch_distinct_values(col_name)
                                value = st.multiselect(f"Values for {col_name}", distinct_values, key=value_key)
                            elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                                value = st.date_input(f"Value for {col_name}", key=value_key)
                            else:
                                distinct_values = fetch_distinct_values(col_name)
                                value = st.selectbox(f"Value for {col_name}", distinct_values, key=value_key)
                
                            return value
                
                        # Initialize variables to store filters and logical conditions
                        filters = []
                        logic_operator = None
                        filter_index = 0
                
                        while True:
                            available_columns = remaining_columns  # Columns available for filtering
                
                            if available_columns.empty:
                                st.write("No more columns available for filtering.")
                                break  # Stop the loop if no columns remain
                
                            # Create 3 columns for column selection, operator, and value input
                            col1, col2, col3 = st.columns([2, 1, 2])
                
                            with col1:
                                selected_column = st.selectbox(f"Column (filter {filter_index + 1})", available_columns)
                
                            # Determine column data type
                            column_info_query = f"""
                                SELECT DATA_TYPE
                                FROM INFORMATION_SCHEMA.COLUMNS
                                WHERE TABLE_NAME = '{tbl}' AND COLUMN_NAME = '{selected_column}';
                            """
                            column_info = session.sql(column_info_query).collect()
                            col_data_type = column_info[0]['DATA_TYPE']  # Get the data type
                
                            with col2:
                                operator = get_operator_input(selected_column, col_data_type, filter_index)
                
                            with col3:
                                value = get_value_input(selected_column, col_data_type, operator, filter_index)
                
                            # Append filter if valid
                            if operator and (value is not None or value == 0):
                                filters.append((selected_column, operator, value))
                
                            # Ask user if they want another filter
                            add_filter = st.radio(f"Add another filter after {selected_column}?", ['No', 'Yes'], key=f"add_filter_{filter_index}")
                
                            if add_filter == 'Yes':
                                col1, col2 = st.columns([2, 13])
                                with col1: 
                                    logic_operator = st.selectbox(f"Choose logical operator after filter {filter_index + 1}", ['AND', 'OR'], key=f"logic_operator_{filter_index}")
                                    filter_index += 1
                                with col2:
                                    st.write("")
                            else:
                                break
                        
                        # Generate SQL WHERE clause based on selected filters and logic
                        sql_where_clause = " AND "
                        #st.write(filters)
                        for i, (col, operator, value) in enumerate(filters):
                            if i > 0 and logic_operator:
                                sql_where_clause += f" {logic_operator} "
                            
                            if operator == 'IN':
                                # Handle IN operator where the value is a list
                                sql_where_clause += f"{col} IN {tuple(value)}"
                            else:
                                # Check if value is numeric
                                    if isinstance(value, (int, float)):
                                        sql_where_clause += f"{col} {operator} {value}"
                                    else:
                                 # For non-numeric values (strings, dates), enclose the value in quotes
                                        sql_where_clause += f"{col} {operator} '{value}'"        
                        else:
                            st.write("")
    if mode == "Union":
        
        with st.expander ("Input Parameters (Compared)"):
            # DATA SOURCE FOR SECOND INSTANCE
            st.markdown("""
            <h2 style='font-size: 14px; margin-bottom: 0px;'>Data Source</h2>
            <hr style='margin-top: -8px;margin-bottom: 5px;'>
            """, unsafe_allow_html=True)
            
             # Get list of databases
            sqldb1 = "SHOW DATABASES"
            databases1 = session.sql(sqldb1).collect()
            db01 = pd.DataFrame(databases1)
            
            col1, col2, col3 = st.columns(3)
            
            # **Database Selection**
            with col1:
                database1 = st.selectbox('Select Database', key='comparecompdb', index=None, 
                                        placeholder="Choose from list...", options=db01['name'].unique())
            
            # **Schema Selection (Only if a database is selected)**
            if database1:
                sqlschemas1 = f"SHOW SCHEMAS IN DATABASE {database1}"
                schemas1 = session.sql(sqlschemas1).collect()
                schema01 = pd.DataFrame(schemas1)
            
                with col2:
                    schema1 = st.selectbox('Select Schema', key='comparecompsch', index=None, 
                                          placeholder="Choose from list...", options=schema01['name'].unique())
            else:
                schema1 = None  # Prevents SQL execution
            
            # **Table Selection (Only if a database & schema are selected)**
            if database1 and schema1:
                sqltables1 = f"""
                    SELECT TABLE_NAME 
                    FROM {database1}.INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = '{schema1}' AND TABLE_TYPE IN ('BASE TABLE', 'VIEW')
                """
                tables1 = session.sql(sqltables1).collect()
                table01 = pd.DataFrame(tables1)
            
                with col3:
                    tbl1 = st.selectbox('Select Event Table or View', key='comparecomptbl', index=None, 
                                       placeholder="Choose from list...", options=table01['TABLE_NAME'].unique(),
                                       help="Select any table or view with events data. Minimum structure should include a unique identifier, an event, and a timestamp. Additional fields can be used to filter data.")
            else:
                tbl1 = None  # Prevents SQL execution
            
            # **Column Selection (Only if a database, schema, and table are selected)**
            if database1 and schema and tbl1:
                cols1 = f"""
                    SELECT COLUMN_NAME
                    FROM {database1}.INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = '{schema1}' AND TABLE_NAME = '{tbl1}'
                    ORDER BY ORDINAL_POSITION;
                """

                colssql1 = session.sql(cols1).collect()
                colsdf1 = pd.DataFrame(colssql1)
            
            col4, col5, col6 = st.columns([4,4,4])
            with col4:
                uid1 = st.selectbox('Select identifier column', colsdf1, index=None,  key='uidcomparecomp',placeholder="Choose from list...", help="The identifier column is the unique identifier for each event record. This may be a user id, a customer id, a process id, etc ...it is used to partition the input set of rows before pattern matching.")
            with col5:
                evt1 = st.selectbox('Select event column', colsdf1, index=None, key='evtcomparecomp', placeholder="Choose from list...",help="The event column contains the actual events that will be used in the path analysis.")
            with col6:
                tmstp1 = st.selectbox('Select timestamp column', colsdf1, index=None, key='tsmtpcomparecomp',placeholder="Choose from list...",help="The timestamp column contains the timestamp associated to the event. This is the value that will be used for sequentially ordering the dataset.")
            
            # Get Distinct Events Of Interest from Event Table
            if (uid1 is not None and evt1 is not None and tmstp1 is not None):
                # Get Distinct Events Of Interest from Event Table
                EOI1 = f"SELECT DISTINCT {evt1} FROM {tbl1} ORDER BY {evt1}"
                # Get start EOI :
                start1 = session.sql(EOI1).collect()
                # Get end EOI :
                end1 = session.sql(EOI1).collect()
                # Get excluded EOI :
                excl1 = session.sql(EOI1).collect()
            
                # Write query output in a pandas dataframe
                startdf1_instance = pd.DataFrame(start1)
                enddf1_instance = pd.DataFrame(end1)
                excl0_instance = pd.DataFrame(excl1)
            
                # Add "any" to the distinct events list
                any_row_instance = {evt1: 'Any'}
            
                # Convert the any_row to a DataFrame and append it using pd.concat
                any_row_df_instance = pd.DataFrame([any_row_instance])
                # Add any to list of FROM
                startdf1_instance_with_any = pd.concat([any_row_df_instance, startdf1_instance], ignore_index=True)
                # Add any to list of TO
                enddf1_instance_with_any = pd.concat([any_row_df_instance, enddf1_instance], ignore_index=True)
            
                # EVENTS PATTERN (Comp)
                with st.container():
                    st.markdown("""
            <h2 style='font-size: 14px; margin-bottom: 0px;'>Events Pattern</h2>
            <hr style='margin-top: -8px;margin-bottom: 5px;'>
            """, unsafe_allow_html=True)
            
                # Add a None placeholder to force user to select an event
                options_with_placeholder_from_instance = ["🔍"] + startdf1_instance_with_any[evt1].unique().tolist()
                options_with_placeholder_to_instance = ["🔎"] + enddf1_instance_with_any[evt1].unique().tolist()
            
                col4, col5, col6, col7 = st.columns([4,4,2,2])
            
                with col4:
                    frm1 = st.multiselect('Select events FROM:', key='evtfromcomp', options=options_with_placeholder_from_instance[1:], default=[], help="Select one or more events of interest to visualize paths FROM the selected point(s). 'Any' matches all values.")
                    if frm1 != "🔍":
                        fromevt1 = ", ".join([f"'{value}'" for value in frm1])
                    else:
                        fromevt1 = None  # Set to None if the placeholder is still selected
            
                with col5:
                    to1 = st.multiselect('Select events TO:',  key='evttocomp', options=options_with_placeholder_to_instance[1:], default=[],help="The event column contains the actual events that will be used in the path analysis.")
                    if to1 != "🔎":
                        toevt1 = ", ".join([f"'{value}'" for value in to1])
                    else:
                        toevt1 = None  # Set to None if the placeholder is still selected
            
                with col6:
                    minnbbevt1 = st.number_input("Min # events", value=0,  key='minnbbevtcomp',placeholder="Type a number...",help="Select the minimum number of events either preceding or following the event(s) of interest.")
            
                with col7:
                    maxnbbevt1 = st.number_input("Max # events", value=5, min_value=1,key='maxnbbevtcomp', placeholder="Type a number...",help="Select the maximum number of events either preceding or following the event(s) of interest.")
            
                col1, col2 = st.columns([5,10])
                with col1:
                    checkoverlap = st.toggle("Allow overlap", key="checkoverlapcompare2",help="Specifies the pattern-matching mode. Toggled-on allows OVERLAP and finds every occurrence of pattern in partition, regardless of whether it is part of a previously found match. One row can match multiple symbols in a given matched pattern.Toggled-off does not allow OVERLAP and starts next pattern search at row that follows last pattern match.")
                    overlap = 'PAST LAST ROW'
                    if checkoverlap:
                        overlap='TO NEXT ROW'
                with col2:
                    st.write("")

                # --------------------------------------
                # DATE RANGE (Comp)
                # --------------------------------------
                with st.container():
                    st.markdown("""
            <h2 style='font-size: 14px; margin-bottom: 0px;'>Date range</h2>
            <hr style='margin-top: -8px;margin-bottom: 5px;'>
            """, unsafe_allow_html=True)
            
                col1, col2, col3 = st.columns([2.4,2.4,10])
            
                # SQL query to get the min start date
                minstartdt1 = f"SELECT TO_VARCHAR(MIN({tmstp1}), 'YYYY/MM/DD') FROM {tbl1}"
                # Get min start date :
                defstartdt1_instance = session.sql(minstartdt1).collect()
                defstartdt1_str_instance = defstartdt1_instance[0][0]
                defstartdt1_date_instance = datetime.datetime.strptime(defstartdt1_str_instance, '%Y/%m/%d').date()
            
                
                with col1:
                    startdt_input1 = st.date_input('Start date', key='startdtcomp',value=defstartdt1_date_instance)
                with col2:
                    enddt_input1  = st.date_input('End date', key='enddtcomp',help="Apply a time window to the data set to find paths on a specific date range or over the entire lifespan of your data (default values)")
                with col3: 
                    st.write("")
            
                # --------------------------------------
                # SESSION (Comp)
                # --------------------------------------
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Session</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)
            
                tab11, tab22 = st.tabs(["Column", "Sessionize"])
               
                with tab11:
                    col1, col2  = st.columns([5,10])
                    with col1:
                            sess1 = st.selectbox('Select session column', colsdf1, index=None, key='sesscomp', placeholder="Choose from list...",help="If a session field is available within the event table.")
                    
                    with col2:
                        st.write("")
                
                with tab22:
                    col1, col2, col3 = st.columns([2.4,2.4,10])
                    with col1:
                        unitoftime1 =st.selectbox( "Unit of time",
                    ("SECOND", "MINUTE", "HOUR","DAY"),index=None, key='unitoftimecomp', placeholder="Choose from list", help="Select the unit of time of the session time window.")
                  
                    with col2:
                        timeout1=  st.number_input( "Insert a timeout value",key='timeoutcomp',value=None, min_value=1, format="%d", placeholder="Type a number",help="Value of the session time window.")
                if sess1 == None and unitoftime1==None and timeout1==None: 
                        partitionby1 = f"partition by {uid1}"
                        groupby1 = f"group by {uid1}, match_number "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns1 = colsdf1[~colsdf1['COLUMN_NAME'].isin([uid1, evt1, tmstp1])]['COLUMN_NAME']
                
                elif sess1 != None and unitoftime1==None and timeout1==None:
                        partitionby1=f"partition by {uid1},{sess1} "
                        groupby1 = f"group by {uid1}, match_number,{sess1} "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns = colsdf1[~colsdf1['COLUMN_NAME'].isin([uid1, evt1, tmstp1,sess1])]['COLUMN_NAME']
    
                elif sess1 == None and unitoftime1 !=None and timeout1 !=None:
                        partitionby1=f"partition by {uid1},SESSION "
                        groupby1 = f"group by {uid1}, match_number,SESSION "
                        # Get the remaining columns after excluding the selected ones
                        remaining_columns1 = colsdf1[~colsdf1['COLUMN_NAME'].isin([uid1, evt1, tmstp1])]['COLUMN_NAME']
                else :
                    st.write("")
            
                # --------------------------------------
                # FILTERS (Comp)
                # --------------------------------------
                #initialize sql_where_clause
                sql_where_clause_instance = ""
                
                with st.container():
                    st.markdown("""<h2 style='font-size: 14px; margin-bottom: 0px;'>Filters</h2><hr style='margin-top: -8px;margin-bottom: 5px;'>""", unsafe_allow_html=True)
            
                col4, col5 = st.columns([5, 10])
                with col4:
                    # Exclude Events
                    excl1 = st.multiselect('Exclude event(s) - optional', excl0_instance, placeholder="Select event(s)...",key='compared',help="Event(s) to be excluded from the pattern evaluation and the ouput.")
            
                    if not excl1:
                        excl3_instance = "''"
                    else:
                        excl3_instance = ', '.join([f"'{excl2}'" for excl2 in excl1])
                # Ensure we have remaining columns before allowing additional filters
                if not remaining_columns.empty:
                    col1, col2 = st.columns([5, 10])
                    with col1:
                        checkfilters1 = st.toggle("Additional filters", key='filterscomparecomp', help="Apply one or many conditional filters to the input data used in the path and pattern analysis.")
                    with col2:
                        st.write("")    
                else:
                    checkfilters1 = False  # Disable filters when no columns are left
                # Only execute filter logic if there are available columns AND the toggle is enabled
                if checkfilters1 and not remaining_columns.empty:
                # ADDITIONAL FILTERS (Comp)
                #checkfilters1 = st.toggle("Additional filters",key='addfilterscomp',help="Apply one or many conditional filters to the input data used in the path and pattern analysis.")
                
                #if checkfilters1:
                    with st.container():
                        # Helper function to fetch distinct values from a column
                        def fetch_distinct_values_instance(column):
                            """Query the distinct values for a column, except for dates"""
                            query = f"SELECT DISTINCT {column} FROM {tbl1}"
                            result = session.sql(query).collect()
                            distinct_values = [row[column] for row in result]
                            return distinct_values
            
                        # Helper function to display operator selection based on column data type
                        def get_operator_input_instance(col_name, col_data_type, filter_index):
                            """ Returns the operator for filtering based on column type """
                            operator_key = f"{col_name}_operator_{filter_index}"  # Ensure unique key
            
                            if col_data_type in ['NUMBER', 'FLOAT', 'INT']:
                                operator = st.selectbox(f"Operator (Comp)", ['=', '<', '<=', '>', '>=', '!=', 'IN'], key=operator_key)
                            elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                                operator = st.selectbox(f"Operator (Comp)", ['<=', '>=', '='], key=operator_key)
                            else:  # For string or categorical columns
                                operator = st.selectbox(f"Operator (Comp)", ['=', '!=', 'IN'], key=operator_key)
                            return operator
            
                        # Helper function to display value input based on column data type
                        def get_value_input_instance(col_name, col_data_type, operator, filter_index):
                            """ Returns the value for filtering based on column type """
                            value_key = f"{col_name}_value_{filter_index}"  # Ensure unique key
            
                            if operator == 'IN':
                                # For IN operator, allow multiple value selection
                                distinct_values = fetch_distinct_values_instance(col_name)
                                value = st.multiselect(f"Values for {col_name} (Comp)", distinct_values, key=value_key)
                            elif col_data_type in ['TIMESTAMP', 'DATE', 'DATETIME', 'TIMESTAMP_NTZ']:
                                # For date columns, let the user input a date value
                                value = st.date_input(f"Value for {col_name} (Comp)", key=value_key)
                            else:
                                # For other operators, allow single value selection from distinct values
                                distinct_values = fetch_distinct_values_instance(col_name)
                                value = st.selectbox(f"Value for {col_name} (Comp)", distinct_values, key=value_key)
            
                            return value
            
                        # Initialize variables to store filters and logical conditions
                        filters_instance = []
                        logic_operator_instance = None
                        filter_index_instance = 0  # A unique index to assign unique keys to each filter
            
                        # Dynamic filter loop with one selectbox at a time
                        while True:
                            available_columns1 = remaining_columns1
            
                            if available_columns1.empty:
                                st.write("No more columns available for filtering (Comp).")
                                break
            
                            # Create columns for column selection, operator, and value input
                            col1, col2, col3 = st.columns([2, 1, 2])
            
                            with col1:
                                selected_column_instance = st.selectbox(f"Column (Filter {filter_index_instance + 1} - Comp)", available_columns1)
            
                            # Determine column data type by querying the INFORMATION_SCHEMA
                            column_info_query1 = f"""
                                SELECT DATA_TYPE
                                FROM INFORMATION_SCHEMA.COLUMNS
                                WHERE TABLE_NAME = '{tbl1}' AND COLUMN_NAME = '{selected_column_instance}';
                            """
                            column_info_instance = session.sql(column_info_query1).collect()
                            col_data_type_instance = column_info_instance[0]['DATA_TYPE']
            
                            with col2:
                                operator_instance = get_operator_input_instance(selected_column_instance, col_data_type_instance, filter_index_instance)
            
                            with col3:
                                value_instance = get_value_input_instance(selected_column_instance, col_data_type_instance, operator_instance, filter_index_instance)
            
                            if operator_instance and (value_instance is not None or value_instance == 0):
                                filters_instance.append((selected_column_instance, operator_instance, value_instance))
            
                            add_filter_instance = st.radio(f"Add another filter after {selected_column_instance} (Comp)?", ['No', 'Yes'], key=f"add_filter_instance_{filter_index_instance}")
            
                            if add_filter_instance == 'Yes':
                                col1, col2 = st.columns([2, 13])
                                with col1:
                                    logic_operator_instance = st.selectbox(f"Choose logical operator after filter {filter_index_instance + 1} (Comp)", ['AND', 'OR'], key=f"logic_operator_instance_{filter_index_instance}")
                                    filter_index_instance += 1
                                with col2:
                                    st.write("")
                            else:
                                break
            
                        # Generate SQL WHERE clause for the second instance
                        sql_where_clause_instance = " AND "
                        for i, (col_instance, operator_instance, value_instance) in enumerate(filters_instance):
                            if i > 0 and logic_operator_instance:
                                sql_where_clause_instance += f" {logic_operator_instance} "
            
                            if operator_instance == 'IN':
                                # Handle IN operator where the value is a list
                                sql_where_clause_instance += f"{col_instance} IN {tuple(value_instance)}"
                            else:
                                # Check if the value is numeric
                                if isinstance(value_instance, (int, float)):
                                    # No need for quotes if the value is numeric
                                    sql_where_clause_instance += f"{col_instance} {operator_instance} {value_instance}"
                                else:
                                    # For non-numeric values (strings, dates), enclose the value in quotes
                                    sql_where_clause_instance += f"{col_instance} {operator_instance} '{value_instance}'"
            
                        # Display the generated SQL WHERE clause
                        #st.write(f"Generated SQL WHERE clause (Comp): {sql_where_clause_instance}")
            
            else:
                st.write("")
        
    # SQL LOGIC
    # Check pattern an run SQL accordingly
    if mode == 'Complement':
        
        if all([uid, evt, tmstp,fromevt, toevt]):
                    #Name Reference and Compared Group
            rename_groups= st.checkbox("Label Reference and Compared groups", key="disabled")
            
            reference_label = 'REFERENCE'
            compared_label = 'COMPARED'
            
            if rename_groups:
                col1, col2 = st.columns(2)
                with col1:
                        reference_label = st.text_input(
                        "Reference Group Name",
                        placeholder="Please name the reference group",
                        )     
                with col2:
                         compared_label = st.text_input(
                         "Compared Group Name",
                        placeholder="Please name the compared group",
                        )
    
            # Continue with SQL generation and execution based on inputs...
    
            # PATH TO: Pattern = A{{{minnbbevt},{maxnbbevt}}} B
            if fromevt.strip("'") == 'Any' and toevt.strip("'") != 'Any':
                crttblrawseventsrefsql= None
                crttblrawseventsref = None
                crttblrawseventscompsql = None
                crttblrawseventscomp = None
                
                if st.button("Compare", key='complementto'):
                    # Generate a unique ref table name
                    def generate_unique_reftable_name(base_name="RAWEVENTSREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftable_name = generate_unique_reftable_name()
                    
                        # CREATE TABLE individiual reference Paths 
                    if unitoftime==None and timeout ==None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (select * from {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}} B) 
                            define A as true, B AS {evt} IN ({toevt})
                        )  {groupby}) """
                    elif unitoftime != None and timeout !=None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                        {tmstp}) AS TIMEWINDOW FROM {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                        ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                        SELECT *FROM sessions) match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}} B) 
                            define A as true, B AS {evt} IN ({toevt})
                        )  {groupby}) """
                        
                    # Run the SQL
                    crttblrawseventsref = session.sql(crttblrawseventsrefsql).collect()
                    # Generate a unique comp table name
                    def generate_unique_comptable_name(base_namec="RAWEVENTSCOMP"):
                        unique_compid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_namec}_{unique_compid}"
                     # Generate a unique table name
                    unique_comptable_name = generate_unique_comptable_name()  
                   
                    
                    # CREATE TABLE individiual compared (complement set) Paths 
                    if unitoftime==None and timeout ==None :
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (select * from {tbl} where {uid} NOT IN (SELECT DISTINCT ({uid}) FROM {unique_reftable_name} ) AND
                        {evt} not in({excl3}) and {tmstp} < (SELECT MAX({tmstp})from {tbl} where {evt} = {toevt} )and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}}) 
                            define A AS {evt} NOT IN ({toevt})
                        )  {groupby}) """
                    elif unitoftime != None and timeout !=None :
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                        {tmstp}) AS TIMEWINDOW FROM {tbl} where  {uid} NOT IN (SELECT DISTINCT ({uid}) FROM {unique_reftable_name} ) AND
                        {evt} not in({excl3}) and {tmstp} < (SELECT MAX({tmstp})from {tbl} where {evt} = {toevt} )and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause})
                        ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                        SELECT *FROM sessions)
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}}) 
                            define A AS {evt} NOT IN ({toevt})
                        )  {groupby}) """
    
                    # Run the SQL
                    crttblrawseventscomp = session.sql(crttblrawseventscompsql).collect()
                    # Generate a unique ref tfidf table name
                    def generate_unique_reftftable_name(base_name="TFIDFREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftftable_name = generate_unique_reftftable_name()
                    
                    #CREATE TABLE TF-IDF Reference
                    crttbltfidfrefsql=f"""CREATE TABLE {unique_reftftable_name} AS
                     (
                        Select
                        {uid},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_reftable_name}, lateral strtok_split_to_table({unique_reftable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfrefsql)
                    crttbltfidfref = session.sql(crttbltfidfrefsql).collect()
                # Generate a unique comp tfidf table name
                    def generate_unique_comptftable_name(base_name="TFIDFCOMP"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_comptftable_name = generate_unique_comptftable_name()
                    
                    #CREATE TABLE TF-IDF Compared
                    crttbltfidfcompsql=f"""CREATE TABLE {unique_comptftable_name} AS
                     (
                        Select
                        {uid},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_comptable_name}, lateral strtok_split_to_table({unique_comptable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfcompsql)
                    crttbltfidfcomp = session.sql(crttbltfidfcompsql).collect()
                    #COMPARE BOTH TFIDF AND RANK 
                    tfidfsql = f"""select 
                    EVENT, grp,
                    rank () over (partition by A.grp order by A.sum_tfidf desc) as ranking,  sum_tfidf as tfidf
                    from
                    (
                    select sum (tfidf) sum_tfidf, event, 'REFERENCE' as grp
                    from {unique_reftftable_name}
                    group by 2,3
                    union ALL
                    select sum (tfidf) sum_tfidf, event, 'COMPARED' as grp
                    from {unique_comptftable_name}
                    group by 2,3) as A"""
                    tfidf = session.sql(tfidfsql).collect()
                    restfidf = pd.DataFrame(tfidf)
                    # Replace "COMPARED" and "REFERENCE" with the dynamic labels
                    restfidf['GRP_Labeled'] = restfidf['GRP'].replace({"COMPARED": compared_label, "REFERENCE": reference_label})
                    
                    # Reshape the data for plotting
                    restfidf_pivot = (
                        restfidf
                        .pivot(index="EVENT", columns="GRP_Labeled", values="RANKING")
                        .reset_index()
                        .melt(id_vars="EVENT", value_name="RANKING", var_name="GRP_Labeled")
                    )
                    
                    # Identify duplicates within each group (GRP_Labeled) and adjust ranking for duplicates only
                    restfidf_pivot['is_duplicate'] = restfidf_pivot.duplicated(subset=['GRP_Labeled', 'RANKING'], keep=False)
                    
                    # Create the adjusted ranking column
                    restfidf_pivot['adjusted_ranking'] = restfidf_pivot['RANKING']
                    restfidf_pivot.loc[restfidf_pivot['is_duplicate'], 'adjusted_ranking'] = (
                        restfidf_pivot.loc[restfidf_pivot['is_duplicate']]
                        .groupby(['GRP_Labeled', 'RANKING'])
                        .cumcount() * 0.3 + restfidf_pivot['RANKING']
                    )
                    
                    # Drop the temporary 'is_duplicate' column if no longer needed
                    restfidf_pivot.drop(columns='is_duplicate', inplace=True)
                    
                    # Calculate dynamic height based on the range of rankings
                    ranking_range = restfidf_pivot["adjusted_ranking"].nunique()  # Number of unique adjusted ranking levels
                    base_height_per_rank = 33  # Adjust this value as needed
                    dynamic_height = ranking_range * base_height_per_rank
                                            
                    # Create the slope chart using Altair with fixed order for GRP_Labeled
                    slope_chart = alt.Chart(restfidf_pivot).mark_line(point=True).encode(
                        x=alt.X('GRP_Labeled:N', title='', sort=[reference_label, compared_label]),
                        y=alt.Y('adjusted_ranking:Q', axis=alt.Axis(grid=False, title='Ranking', labelAngle=0), sort='descending'),
                        color=alt.Color('EVENT:N', legend=alt.Legend(title="Event", labelFontSize=8, titleFontSize=10)),
                        detail='EVENT:N',
                        tooltip=[
                            alt.Tooltip('RANKING:Q', title="Original Ranking"),
                            alt.Tooltip('EVENT:N', title="Event")
                        ]
                    ).properties(
                        width=800,
                        height=dynamic_height,
                        title=""
                    )
                    
                    # Add labels for each point
                    # For the COMPARED group (right-side labels)
                    text_compared = alt.Chart(restfidf_pivot).mark_text(align='left', dx=5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', title='', axis=alt.Axis(labelAngle=0), sort=[reference_label, compared_label]),
                        y=alt.Y('adjusted_ranking:Q', sort='descending'),
                        text=alt.Text('EVENT:N'),
                        color=alt.Color('EVENT:N'),
                        tooltip=[
                            alt.Tooltip('RANKING:Q', title="Original Ranking"),
                            alt.Tooltip('EVENT:N', title="Event")
                        ]
                    ).transform_filter(
                        alt.datum.GRP_Labeled == compared_label
                    )
                    
                    # For the REFERENCE group (left-side labels)
                    text_reference = alt.Chart(restfidf_pivot).mark_text(align='right', dx=-5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', title='', axis=alt.Axis(labelAngle=0),sort=[reference_label, compared_label]),
                        y=alt.Y('adjusted_ranking:Q', sort='descending'),
                        text=alt.Text('EVENT:N'),
                        color=alt.Color('EVENT:N'),
                        tooltip=[
                            alt.Tooltip('RANKING:Q', title="Original Ranking"),
                            alt.Tooltip('EVENT:N', title="Event")
                        ]
                    ).transform_filter(
                        alt.datum.GRP_Labeled == reference_label
                    )
                    
           
                    
                    # Combine chart and text
                    final_chart = slope_chart + text_compared + text_reference 
                    
                    # Display in Streamlit
                    st.altair_chart(final_chart, use_container_width=True)
                                        
                    drptblraweventsrefsql =f"""DROP TABLE IF EXISTS {unique_reftable_name}"""
                    drptblraweventsref = session.sql(drptblraweventsrefsql).collect()
                    
                    drptblraweventscompsql =f"""DROP TABLE IF EXISTS {unique_comptable_name}"""
                    drptblraweventscomp = session.sql(drptblraweventscompsql).collect()
                     
                    drptblcomptfsql = f"""DROP TABLE IF EXISTS {unique_comptftable_name}"""
                    drptblcomptf = session.sql(drptblcomptfsql).collect()

                    drptblreftfsql = f"""DROP TABLE IF EXISTS {unique_reftftable_name}"""
                    drptblreftf = session.sql(drptblreftfsql).collect()
                else:
                        st.write("") 
    
            # Separate block for PATH FROM 
            elif fromevt.strip("'") != 'Any' and toevt.strip("'")== 'Any':
                crttblrawseventsrefsql= None
                crttblrawseventsref = None
                crttblrawseventscompsql = None
                crttblrawseventscomp = None
    
                if st.button("Compare", key='complementfrom'):
                       # Generate a unique ref table name
                    def generate_unique_reftable_name(base_name="RAWEVENTSREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftable_name = generate_unique_reftable_name()
                         # CREATE TABLE individiual reference Paths 
                    if unitoftime==None and timeout ==None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (select * from {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A B{{{minnbbevt},{maxnbbevt}}}) 
                            define B as true, A AS {evt} IN ({fromevt})
                        )  {groupby}) """
                        st.write(crttblrawseventsrefsql)
                    
                    elif unitoftime != None and timeout !=None :
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                        {tmstp}) AS TIMEWINDOW FROM {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                        ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                        SELECT *FROM sessions)
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A B{{{minnbbevt},{maxnbbevt}}}) 
                            define B as true, A AS {evt} IN ({fromevt})
                        )  {groupby}) """
                        st.write(crttblrawseventsrefsql)
                    # Run the SQL
                    crttblrawseventsref = session.sql(crttblrawseventsrefsql).collect()
                     # Generate a unique comp table name
                    def generate_unique_comptable_name(base_namec="RAWEVENTSCOMP"):
                        unique_compid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_namec}_{unique_compid}"
                     # Generate a unique table name
                    unique_comptable_name = generate_unique_comptable_name()  
                   
                    
                    # CREATE TABLE individiual compared (complement set) Paths 
                    if unitoftime==None and timeout ==None :
                    
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (select * from {tbl} where {uid} NOT IN (SELECT DISTINCT ({uid}) FROM {unique_reftable_name} ) AND
                        {evt} not in({excl3})and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}}) 
                            define A AS {evt} NOT IN ({fromevt})
                        )  {groupby}) """
                       
                    elif unitoftime != None and timeout !=None :
                        
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                        {tmstp}) AS TIMEWINDOW FROM {tbl} where  {uid} NOT IN (SELECT DISTINCT ({uid}) FROM {unique_reftable_name} ) AND
                        {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause})
                        ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                        SELECT *FROM sessions)
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}}) 
                            define A AS {evt} NOT IN ({fromevt})
                        )  {groupby}) """
                        
                    # Run the SQL
                    crttblrawseventscomp = session.sql(crttblrawseventscompsql).collect()
            
                    # Generate a unique ref tfidf table name
                    def generate_unique_reftftable_name(base_name="TFIDFREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftftable_name = generate_unique_reftftable_name()
                    
                    #CREATE TABLE TF-IDF Reference
                    crttbltfidfrefsql=f"""CREATE TABLE {unique_reftftable_name} AS
                     (
                        Select
                        {uid},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_reftable_name}, lateral strtok_split_to_table({unique_reftable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfrefsql)
                    crttbltfidfref = session.sql(crttbltfidfrefsql).collect()
                # Generate a unique comp tfidf table name
                    def generate_unique_comptftable_name(base_name="TFIDFCOMP"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_comptftable_name = generate_unique_comptftable_name()
                    
                    #CREATE TABLE TF-IDF Compared
                    crttbltfidfcompsql=f"""CREATE TABLE {unique_comptftable_name} AS
                     (
                        Select
                        {uid},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_comptable_name}, lateral strtok_split_to_table({unique_comptable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfcompsql)
                    crttbltfidfcomp = session.sql(crttbltfidfcompsql).collect()
                    #COMPARE BOTH TFIDF AND RANK 
                    tfidfsql = f"""select 
                    EVENT, grp,
                    rank () over (partition by A.grp order by A.sum_tfidf desc) as ranking,  sum_tfidf as tfidf
                    from
                    (
                    select sum (tfidf) sum_tfidf, event, 'REFERENCE' as grp
                    from {unique_reftftable_name}
                    group by 2,3
                    union ALL
                    select sum (tfidf) sum_tfidf, event, 'COMPARED' as grp
                    from {unique_comptftable_name}
                    group by 2,3) as A"""
                    tfidf = session.sql(tfidfsql).collect()
                    restfidf = pd.DataFrame(tfidf)
                    # Replace "COMPARED" and "REFERENCE" with the dynamic labels
                    restfidf['GRP_Labeled'] = restfidf['GRP'].replace({"COMPARED": compared_label, "REFERENCE": reference_label})
                   
                # Reshape the data for plotting
                    restfidf_pivot = restfidf.pivot(index="EVENT", columns="GRP_Labeled", values="RANKING").reset_index().melt(id_vars="EVENT", value_name="RANKING",var_name="GRP_Labeled")
                # Calculate dynamic height based on the range of rankings
                    ranking_range = restfidf_pivot["RANKING"].nunique()  # Number of unique ranking levels
                    base_height_per_rank = 33  # Adjust this value as needed
                    dynamic_height = ranking_range * base_height_per_rank
                    
                    # Add adjusted ranking to avoid overlap
                    restfidf_pivot['adjusted_ranking'] = restfidf_pivot['RANKING'] + restfidf_pivot.groupby('RANKING').cumcount() * 0.3
                    
                    # Create the slope chart using Altair with fixed order for GRP_Labeled
                    slope_chart = alt.Chart(restfidf_pivot).mark_line(point=True).encode(
                        x=alt.X('GRP_Labeled:N', title='', axis=alt.Axis(labelAngle=0), sort=[reference_label, compared_label]),  # Fixed order
                        y=alt.Y('adjusted_ranking:Q', scale=alt.Scale(reverse=True), axis=alt.Axis(grid=False), title='Ranking'),
                        color=alt.Color('EVENT:N', legend=alt.Legend(title="Event", labelFontSize=8, titleFontSize=10)),
                        detail='EVENT:N',
                        tooltip=[alt.Tooltip('RANKING:Q', title="Original Ranking")]  # Tooltip for non-adjusted ranking
                    ).properties(
                        width=800,
                        height=dynamic_height,
                        title=""
                    )
                    
                    # Add labels for each point
                    # For the "COMPARED" group, we place labels on the right
                    text_compared = alt.Chart(restfidf_pivot).mark_text(align='left', dx=5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', sort=[reference_label, compared_label]),  # Fixed order
                        y=alt.Y('adjusted_ranking:Q'),
                        text=alt.Text('EVENT:N'),  # Explicitly set text to EVENT column
                        color=alt.Color('EVENT:N')
                    ).transform_filter(
                        alt.datum.GRP_Labeled == compared_label
                    )
                    
                    # For the "REFERENCE" group, we place labels on the left
                    text_reference = alt.Chart(restfidf_pivot).mark_text(align='right', dx=-5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', sort=[reference_label, compared_label]),  # Fixed order
                        y=alt.Y('adjusted_ranking:Q'),
                        text=alt.Text('EVENT:N'),
                        color=alt.Color('EVENT:N')
                    ).transform_filter(
                        alt.datum.GRP_Labeled == reference_label
                    )
                    
                    # Combine chart and text
                    final_chart = slope_chart + text_compared + text_reference
                    
                    # Display in Streamlit
                    st.altair_chart(final_chart, use_container_width=True)
                                        
                    drptblraweventsrefsql =f"""DROP TABLE IF EXISTS {unique_reftable_name}"""
                    drptblraweventsref = session.sql(drptblraweventsrefsql).collect()
                    
                    drptblraweventscompsql =f"""DROP TABLE IF EXISTS {unique_comptable_name}"""
                    drptblraweventscomp = session.sql(drptblraweventscompsql).collect()
                     
                    drptblcomptfsql = f"""DROP TABLE IF EXISTS {unique_comptftable_name}"""
                    drptblcomptf = session.sql(drptblcomptfsql).collect()

                    drptblreftfsql = f"""DROP TABLE IF EXISTS {unique_reftftable_name}"""
                    drptblreftf = session.sql(drptblreftfsql).collect()
        
                else:
                    st.write("")
                    
            # Separate block for PATH BETWEEN 
            elif fromevt.strip("'") != 'Any' and toevt.strip("'") != 'Any':
                st.warning("Not a valid pattern for comparison")
    
            elif fromevt.strip("'") == 'Any' and toevt.strip("'") == 'Any':
                st.warning("This is tuple generator - Not a valid pattern for comparison")
                
            else:
                st.write("Please select appropriate options for 'from' and 'to'")
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
                    Please ensure all required inputs are selected before running the app. Valid patterns for comparison are 'path to' (FROM= Any) and 'path from' (TO = Any)
                </h5>
            </div>
            """, unsafe_allow_html=True)
            #st.warning("Please ensure all required inputs are selected before running the app. Valid patterns for comparison are 'path to' (FROM= Any) and 'path from' (TO = Any)")
   
    elif mode == 'Union':
        if all([uid, evt, tmstp,fromevt, toevt,uid1, evt1, tmstp1,fromevt1, toevt1]):
                    #Name Reference and Compared Group
            rename_groups= st.checkbox("Label Reference and Compared groups", key="disabled")
            
            reference_label = 'REFERENCE'
            compared_label = 'COMPARED'
            
            if rename_groups:
                col1, col2 = st.columns(2)
                with col1:
                        reference_label = st.text_input(
                        "Reference Group Name",
                        placeholder="Please name the reference group",
                        )     
                with col2:
                         compared_label = st.text_input(
                         "Compared Group Name",
                        placeholder="Please name the compared group",
                        )
            # PATH TO: Pattern = A{{{minnbbevt},{maxnbbevt}}} B
            if fromevt.strip("'") == 'Any' and toevt.strip("'") != 'Any' and fromevt1.strip("'") == 'Any' and toevt1.strip("'") != 'Any':
                crttblrawseventsrefsql= None
                crttblrawseventsref = None
                crttblrawseventscompsql = None
                crttblrawseventscomp = None
                
                if st.button("Compare", key='unionto'):
                    # Generate a unique ref table name
                    def generate_unique_reftable_name(base_name="RAWEVENTSREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftable_name = generate_unique_reftable_name()
                    
                        # CREATE TABLE individiual reference Paths 
                    if unitoftime==None and timeout ==None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (select * from {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}} B) 
                            define A as true, B AS {evt} IN ({toevt})
                        )  {groupby}) """
                    elif unitoftime != None and timeout !=None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                        {tmstp}) AS TIMEWINDOW FROM {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                        ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                        SELECT *FROM sessions) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A{{{minnbbevt},{maxnbbevt}}} B) 
                            define A as true, B AS {evt} IN ({toevt})
                        )  {groupby}) """
                    # Run the SQL
                    crttblrawseventsref = session.sql(crttblrawseventsrefsql).collect()
                    # Generate a unique comp table name
                    def generate_unique_comptable_name(base_namec="RAWEVENTSCOMP"):
                        unique_compid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_namec}_{unique_compid}"
                     # Generate a unique table name
                    unique_comptable_name = generate_unique_comptable_name()  
                   
                    
                    # CREATE TABLE individiual compared (complement set) Paths 
                    if unitoftime1==None and timeout1 ==None :
                        
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid1}, listagg({evt1}, ',') within group (order by MSQ) as path
                        from  (select * from {tbl1} where  {evt1} not in({excl3_instance}) and {tmstp1} between DATE('{startdt_input1}') and DATE('{enddt_input1}') {sql_where_clause_instance}) 
                            match_recognize(
                            {partitionby1} 
                            order by {tmstp1} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap1}
                            pattern(A{{{minnbbevt1},{maxnbbevt1}}} B) 
                            define A as true, B AS {evt1} IN ({toevt1})
                        )  {groupby1}) """
                    elif unitoftime1 != None and timeout1 !=None :
                        
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid1}, listagg({evt1}, ',') within group (order by MSQ) as path
                        from  (WITH events_with_diff AS ( SELECT {uid1},{tmstp1},{evt1},TIMESTAMPDIFF({unitoftime1}, LAG({tmstp1}) OVER (PARTITION BY  {uid1} ORDER BY {tmstp1}),
                        {tmstp1}) AS TIMEWINDOW FROM {tbl1} where  {evt1} not in({excl3_instance}) and {tmstp1} between DATE('{startdt_input1}') and DATE('{enddt_input1}'){sql_where_clause_instance})
                        ,sessions AS (SELECT {uid1},{tmstp1},{evt1},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout1} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid1} ORDER BY {tmstp1}) AS session FROM events_with_diff)
                        SELECT *FROM sessions) 
                            match_recognize(
                            {partitionby1} 
                            order by {tmstp1} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap1}
                            pattern(A{{{minnbbevt1},{maxnbbevt1}}} B) 
                            define A as true, B AS {evt1} IN ({toevt1})
                        )  {groupby1}) """
                        
                    # Run the SQL
                    crttblrawseventscomp = session.sql(crttblrawseventscompsql).collect()
                    # Generate a unique ref tfidf table name
                    def generate_unique_reftftable_name(base_name="TFIDFREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftftable_name = generate_unique_reftftable_name()
                    
                    #CREATE TABLE TF-IDF Reference
                    crttbltfidfrefsql=f"""CREATE TABLE {unique_reftftable_name} AS
                     (
                        Select
                        {uid},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_reftable_name}, lateral strtok_split_to_table({unique_reftable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfrefsql)
                    crttbltfidfref = session.sql(crttbltfidfrefsql).collect()
                # Generate a unique comp tfidf table name
                    def generate_unique_comptftable_name(base_name="TFIDFCOMP"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_comptftable_name = generate_unique_comptftable_name()
                    
                    #CREATE TABLE TF-IDF Compared
                    crttbltfidfcompsql=f"""CREATE TABLE {unique_comptftable_name} AS
                     (
                        Select
                        {uid1},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid1}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid1}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid1}, SEQ) OVER () /count(DISTINCT {uid1},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid1}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid1}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid1}, SEQ) OVER () /count(DISTINCT {uid1},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_comptable_name}, lateral strtok_split_to_table({unique_comptable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfcompsql)
                    crttbltfidfcomp = session.sql(crttbltfidfcompsql).collect()
                    #COMPARE BOTH TFIDF AND RANK 
                    tfidfsql = f"""select 
                    EVENT, grp,
                    rank () over (partition by A.grp order by A.sum_tfidf desc) as ranking,  sum_tfidf as tfidf
                    from
                    (
                    select sum (tfidf) sum_tfidf, event, 'REFERENCE' as grp
                    from {unique_reftftable_name}
                    group by 2,3
                    union ALL
                    select sum (tfidf) sum_tfidf, event, 'COMPARED' as grp
                    from {unique_comptftable_name}
                    group by 2,3) as A"""
                    tfidf = session.sql(tfidfsql).collect()
                    restfidf = pd.DataFrame(tfidf)
                    # Replace "COMPARED" and "REFERENCE" with the dynamic labels
                    restfidf['GRP_Labeled'] = restfidf['GRP'].replace({"COMPARED": compared_label, "REFERENCE": reference_label})
                   
                    # Replace "COMPARED" and "REFERENCE" with the dynamic labels
                    restfidf['GRP_Labeled'] = restfidf['GRP'].replace({"COMPARED": compared_label, "REFERENCE": reference_label})
                    
                    # Reshape the data for plotting
                    restfidf_pivot = (
                        restfidf
                        .pivot(index="EVENT", columns="GRP_Labeled", values="RANKING")
                        .reset_index()
                        .melt(id_vars="EVENT", value_name="RANKING", var_name="GRP_Labeled")
                    )
                    
                    # Identify duplicates within each group (GRP_Labeled) and adjust ranking for duplicates only
                    restfidf_pivot['is_duplicate'] = restfidf_pivot.duplicated(subset=['GRP_Labeled', 'RANKING'], keep=False)
                    
                    # Create the adjusted ranking column
                    restfidf_pivot['adjusted_ranking'] = restfidf_pivot['RANKING']
                    restfidf_pivot.loc[restfidf_pivot['is_duplicate'], 'adjusted_ranking'] = (
                        restfidf_pivot.loc[restfidf_pivot['is_duplicate']]
                        .groupby(['GRP_Labeled', 'RANKING'])
                        .cumcount() * 0.3 + restfidf_pivot['RANKING']
                    )
                    
                    # Drop the temporary 'is_duplicate' column if no longer needed
                    restfidf_pivot.drop(columns='is_duplicate', inplace=True)
                    
                    # Calculate dynamic height based on the range of rankings
                    ranking_range = restfidf_pivot["adjusted_ranking"].nunique()  # Number of unique adjusted ranking levels
                    base_height_per_rank = 33  # Adjust this value as needed
                    dynamic_height = ranking_range * base_height_per_rank
                                            
                    # Create the slope chart using Altair with fixed order for GRP_Labeled
                    slope_chart = alt.Chart(restfidf_pivot).mark_line(point=True).encode(
                        x=alt.X('GRP_Labeled:N', title='', sort=[reference_label, compared_label]),
                        y=alt.Y('adjusted_ranking:Q', axis=alt.Axis(grid=False, title='Ranking', labelAngle=0), sort='descending'),
                        color=alt.Color('EVENT:N', legend=alt.Legend(title="Event", labelFontSize=8, titleFontSize=10)),
                        detail='EVENT:N',
                        tooltip=[
                            alt.Tooltip('RANKING:Q', title="Original Ranking"),
                            alt.Tooltip('EVENT:N', title="Event")
                        ]
                    ).properties(
                        width=800,
                        height=dynamic_height,
                        title=""
                    )
                    
                    # Add labels for each point
                    # For the COMPARED group (right-side labels)
                    text_compared = alt.Chart(restfidf_pivot).mark_text(align='left', dx=5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', title='', axis=alt.Axis(labelAngle=0), sort=[reference_label, compared_label]),
                        y=alt.Y('adjusted_ranking:Q', sort='descending'),
                        text=alt.Text('EVENT:N'),
                        color=alt.Color('EVENT:N'),
                        tooltip=[
                            alt.Tooltip('RANKING:Q', title="Original Ranking"),
                            alt.Tooltip('EVENT:N', title="Event")
                        ]
                    ).transform_filter(
                        alt.datum.GRP_Labeled == compared_label
                    )
                    
                    # For the REFERENCE group (left-side labels)
                    text_reference = alt.Chart(restfidf_pivot).mark_text(align='right', dx=-5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', title='', axis=alt.Axis(labelAngle=0),sort=[reference_label, compared_label]),
                        y=alt.Y('adjusted_ranking:Q', sort='descending'),
                        text=alt.Text('EVENT:N'),
                        color=alt.Color('EVENT:N'),
                        tooltip=[
                            alt.Tooltip('RANKING:Q', title="Original Ranking"),
                            alt.Tooltip('EVENT:N', title="Event")
                        ]
                    ).transform_filter(
                        alt.datum.GRP_Labeled == reference_label
                    )
                    
           
                    
                    # Combine chart and text
                    final_chart = slope_chart + text_compared + text_reference 
                    
                    # Display in Streamlit
                    st.altair_chart(final_chart, use_container_width=True)
                                        
                    drptblraweventsrefsql =f"""DROP TABLE IF EXISTS {unique_reftable_name}"""
                    drptblraweventsref = session.sql(drptblraweventsrefsql).collect()
                    
                    drptblraweventscompsql =f"""DROP TABLE IF EXISTS {unique_comptable_name}"""
                    drptblraweventscomp = session.sql(drptblraweventscompsql).collect()
                     
                    drptblcomptfsql = f"""DROP TABLE IF EXISTS {unique_comptftable_name}"""
                    drptblcomptf = session.sql(drptblcomptfsql).collect()

                    drptblreftfsql = f"""DROP TABLE IF EXISTS {unique_reftftable_name}"""
                    drptblreftf = session.sql(drptblreftfsql).collect()
                else:
                        st.write("") 
    
            # Separate block for PATH FROM 
            elif fromevt.strip("'") != 'Any' and toevt.strip("'")== 'Any'and fromevt1.strip("'") != 'Any' and toevt1.strip("'")== 'Any':
                crttblrawseventsrefsql= None
                crttblrawseventsref = None
                crttblrawseventscompsql = None
                crttblrawseventscomp = None
                
                if st.button("Compare", key='unionfrom'):
                       # Generate a unique ref table name
                    def generate_unique_reftable_name(base_name="RAWEVENTSREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftable_name = generate_unique_reftable_name()
                         # CREATE TABLE individiual reference Paths 
                    if unitoftime==None and timeout ==None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from  (select * from {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}') {sql_where_clause}) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A B{{{minnbbevt},{maxnbbevt}}}) 
                            define B as true, A AS {evt} IN ({fromevt})
                        )  {groupby}) """
                    elif unitoftime != None and timeout !=None :
                        
                        crttblrawseventsrefsql = f"""CREATE TABLE {unique_reftable_name} AS (
                        select {uid}, listagg({evt}, ',') within group (order by MSQ) as path
                        from (WITH events_with_diff AS ( SELECT {uid},{tmstp},{evt},TIMESTAMPDIFF({unitoftime}, LAG({tmstp}) OVER (PARTITION BY  {uid} ORDER BY {tmstp}),
                        {tmstp}) AS TIMEWINDOW FROM {tbl} where  {evt} not in({excl3}) and {tmstp} between DATE('{startdt_input}') and DATE('{enddt_input}'){sql_where_clause})
                        ,sessions AS (SELECT {uid},{tmstp},{evt},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid} ORDER BY {tmstp}) AS session FROM events_with_diff)
                        SELECT *FROM sessions) 
                            match_recognize(
                            {partitionby} 
                            order by {tmstp} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap}
                            pattern(A B{{{minnbbevt},{maxnbbevt}}}) 
                            define B as true, A AS {evt} IN ({fromevt})
                        )  {groupby}) """
                        
                        
                    crttblrawseventsref = session.sql(crttblrawseventsrefsql).collect()
                     # Generate a unique comp table name
                    def generate_unique_comptable_name(base_namec="RAWEVENTSCOMP"):
                        unique_compid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_namec}_{unique_compid}"
                     # Generate a unique table name
                    unique_comptable_name = generate_unique_comptable_name()  
                   
                    
                    # CREATE TABLE individiual compared (complement set) Paths 
                    if unitoftime1==None and timeout1 ==None :
                        
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid1}, listagg({evt1}, ',') within group (order by MSQ) as path
                        from  (select * from {tbl1} where  {evt1} not in({excl3_instance}) and {tmstp1} between DATE('{startdt_input1}') and DATE('{enddt_input1}') {sql_where_clause_instance}) 
                            match_recognize(
                            {partitionby1} 
                            order by {tmstp1} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap1}
                            pattern(A B{{{minnbbevt1},{maxnbbevt1}}}) 
                            define B as true, A AS {evt1} IN ({fromevt1})
                        )  {groupby1}) """
                    elif unitoftime1 != None and timeout1 !=None :
                        
                        crttblrawseventscompsql = f"""CREATE TABLE {unique_comptable_name} AS (
                        select {uid1}, listagg({evt1}, ',') within group (order by MSQ) as path
                        from (WITH events_with_diff AS ( SELECT {uid1},{tmstp1},{evt1},TIMESTAMPDIFF({unitoftime1}, LAG({tmstp1}) OVER (PARTITION BY  {uid1} ORDER BY {tmstp1}),
                        {tmstp1}) AS TIMEWINDOW FROM {tbl1} where  {evt1} not in({excl3_instance}) and {tmstp1} between DATE('{startdt_input1}') and DATE('{enddt_input1}'){sql_where_clause_instance})
                        ,sessions AS (SELECT {uid1},{tmstp1},{evt1},TIMEWINDOW, SUM(CASE WHEN TIMEWINDOW > {timeout1} OR TIMEWINDOW IS NULL THEN 1 ELSE 0 END)
                        OVER (PARTITION BY {uid1} ORDER BY {tmstp1}) AS session FROM events_with_diff)
                        SELECT *FROM sessions) 
                            match_recognize(
                            {partitionby1} 
                            order by {tmstp1} 
                            measures match_number() as MATCH_NUMBER, match_sequence_number() as msq, classifier() as cl 
                            all rows per match
                            AFTER MATCH SKIP {overlap1}
                            pattern(A B{{{minnbbevt1},{maxnbbevt1}}}) 
                            define B as true, A AS {evt1} IN ({fromevt1})
                        )  {groupby1}) """
                        
                    # Run the SQL
                    crttblrawseventscomp = session.sql(crttblrawseventscompsql).collect()
                    # Generate a unique ref tfidf table name
                    def generate_unique_reftftable_name(base_name="TFIDFREF"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_reftftable_name = generate_unique_reftftable_name()
                    
                    #CREATE TABLE TF-IDF Reference
                    crttbltfidfrefsql=f"""CREATE TABLE {unique_reftftable_name} AS
                     (
                        Select
                        {uid},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid}, SEQ) OVER () /count(DISTINCT {uid},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_reftable_name}, lateral strtok_split_to_table({unique_reftable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfrefsql)
                    crttbltfidfref = session.sql(crttbltfidfrefsql).collect()
                # Generate a unique comp tfidf table name
                    def generate_unique_comptftable_name(base_name="TFIDFCOMP"):
                        unique_refid = uuid.uuid4().hex  # Generate a random UUID
                        return f"{base_name}_{unique_refid}"
                    unique_comptftable_name = generate_unique_comptftable_name()
                    
                    #CREATE TABLE TF-IDF Compared
                    crttbltfidfcompsql=f"""CREATE TABLE {unique_comptftable_name} AS
                     (
                        Select
                        {uid1},SEQ,INDEX,
                        count(1) OVER (PARTITION BY {uid1}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid1}, SEQ) as TF,
                        LOG (10,COUNT(DISTINCT {uid1}, SEQ) OVER () /count(DISTINCT {uid1},SEQ) OVER (PARTITION BY VALUE)) as IDF,
                        (count(1) OVER (PARTITION BY {uid1}, SEQ, VALUE)/ count(1) OVER (PARTITION BY {uid1}, SEQ) )*(LOG (10,COUNT(DISTINCT {uid1}, SEQ) OVER () /count(DISTINCT {uid1},SEQ) OVER (PARTITION BY VALUE))) AS TFIDF,
                       VALUE AS EVENT
                        FROM
                        (select * 
                        from {unique_comptable_name}, lateral strtok_split_to_table({unique_comptable_name}.path, ',')
                        order by seq, index)
                        )"""
                    
                    #st.write(crttbltfidfcompsql)
                    crttbltfidfcomp = session.sql(crttbltfidfcompsql).collect()
                    #COMPARE BOTH TFIDF AND RANK 
                    tfidfsql = f"""select 
                    EVENT, grp,
                    rank () over (partition by A.grp order by A.sum_tfidf desc) as ranking,  sum_tfidf as tfidf
                    from
                    (
                    select sum (tfidf) sum_tfidf, event, 'REFERENCE' as grp
                    from {unique_reftftable_name}
                    group by 2,3
                    union ALL
                    select sum (tfidf) sum_tfidf, event, 'COMPARED' as grp
                    from {unique_comptftable_name}
                    group by 2,3) as A"""
                    tfidf = session.sql(tfidfsql).collect()
                    restfidf = pd.DataFrame(tfidf)
                    # Replace "COMPARED" and "REFERENCE" with the dynamic labels
                    restfidf['GRP_Labeled'] = restfidf['GRP'].replace({"COMPARED": compared_label, "REFERENCE": reference_label})
                   
                # Reshape the data for plotting
                    restfidf_pivot = restfidf.pivot(index="EVENT", columns="GRP_Labeled", values="RANKING").reset_index().melt(id_vars="EVENT", value_name="RANKING",var_name="GRP_Labeled")
                # Calculate dynamic height based on the range of rankings
                    ranking_range = restfidf_pivot["RANKING"].nunique()  # Number of unique ranking levels
                    base_height_per_rank = 33  # Adjust this value as needed
                    dynamic_height = ranking_range * base_height_per_rank
                    
                    # Add adjusted ranking to avoid overlap
                    restfidf_pivot['adjusted_ranking'] = restfidf_pivot['RANKING'] + restfidf_pivot.groupby('RANKING').cumcount() * 0.3
                    
                    # Create the slope chart using Altair with fixed order for GRP_Labeled
                    slope_chart = alt.Chart(restfidf_pivot).mark_line(point=True).encode(
                        x=alt.X('GRP_Labeled:N', title='', axis=alt.Axis(labelAngle=0), sort=[reference_label, compared_label]),  # Fixed order
                        y=alt.Y('adjusted_ranking:Q', scale=alt.Scale(reverse=True), axis=alt.Axis(grid=False), title='Ranking'),
                        color=alt.Color('EVENT:N', legend=alt.Legend(title="Event", labelFontSize=8, titleFontSize=10)),
                        detail='EVENT:N',
                        tooltip=[alt.Tooltip('RANKING:Q', title="Original Ranking")]  # Tooltip for non-adjusted ranking
                    ).properties(
                        width=800,
                        height=dynamic_height,
                        title=""
                    )
                    
                    # Add labels for each point
                    # For the "COMPARED" group, we place labels on the right
                    text_compared = alt.Chart(restfidf_pivot).mark_text(align='left', dx=5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', sort=[reference_label, compared_label]),  # Fixed order
                        y=alt.Y('adjusted_ranking:Q'),
                        text=alt.Text('EVENT:N'),  # Explicitly set text to EVENT column
                        color=alt.Color('EVENT:N')
                    ).transform_filter(
                        alt.datum.GRP_Labeled == compared_label
                    )
                    
                    # For the "REFERENCE" group, we place labels on the left
                    text_reference = alt.Chart(restfidf_pivot).mark_text(align='right', dx=-5, fontSize=10).encode(
                        x=alt.X('GRP_Labeled:N', sort=[reference_label, compared_label]),  # Fixed order
                        y=alt.Y('adjusted_ranking:Q'),
                        text=alt.Text('EVENT:N'),
                        color=alt.Color('EVENT:N')
                    ).transform_filter(
                        alt.datum.GRP_Labeled == reference_label
                    )
                    
                    # Combine chart and text
                    final_chart = slope_chart + text_compared + text_reference
                    
                    # Display in Streamlit
                    st.altair_chart(final_chart, use_container_width=True)
                                        
                    drptblraweventsrefsql =f"""DROP TABLE IF EXISTS {unique_reftable_name}"""
                    drptblraweventsref = session.sql(drptblraweventsrefsql).collect()
                    
                    drptblraweventscompsql =f"""DROP TABLE IF EXISTS {unique_comptable_name}"""
                    drptblraweventscomp = session.sql(drptblraweventscompsql).collect()
                     
                    drptblcomptfsql = f"""DROP TABLE IF EXISTS {unique_comptftable_name}"""
                    drptblcomptf = session.sql(drptblcomptfsql).collect()

                    drptblreftfsql = f"""DROP TABLE IF EXISTS {unique_reftftable_name}"""
                    drptblreftf = session.sql(drptblreftfsql).collect()
        
                else:
                    st.write("")
                    
            # Separate block for PATH BETWEEN 
            elif fromevt.strip("'") != 'Any' and toevt.strip("'") != 'Any' and fromevt1.strip("'") != 'Any' and toevt1.strip("'") != 'Any':
                st.warning("Not a valid pattern for comparison")
    
            elif fromevt.strip("'") == 'Any' and toevt.strip("'") == 'Any' and fromevt1.strip("'") == 'Any' and toevt1.strip("'") == 'Any':
                st.warning("This is tuple generator - Not a valid pattern for comparison")
                
            else:
                st.write("Please select appropriate options for 'from' and 'to'")
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
                    Please ensure all required inputs are selected before running the app. Valid patterns for comparison are 'path to' (FROM= Any) and 'path from' (TO = Any)
                </h5>
            </div>
            """, unsafe_allow_html=True)
            #st.warning("Please ensure all required inputs are selected before running the app. Valid patterns for comparison are 'path to' (FROM= Any) and 'path from' (TO = Any)")