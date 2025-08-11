# Snowpath

Snowpath™ native Streamlit application allows users to easily and visually perform and deep dive into Path Analysis, Attribution Analysis, Association Analysis and event-based Prediction Modeling by simply specifying a few parameters in drop-down menus. Leveraging advanced techniques, Snowpath™ intuitively and visually helps identify touchpoints influencing customer (or machine) behaviours, targets them to create segments, performs cross-population behavioural comparisons, computes rule-based and ML-driven attribution models to understand the contribution of each event preceding a specific outcome, conducts association analysis to uncover hidden patterns and relationships between events and eventually predict outcomes based on event sequences. Snowpath™ is designed to be user-friendly and accessible to users with varying levels of technical expertise, making it a valuable tool for data-driven decision-making. Snowpath™ also leverages the interpretive and generative power of LLMs thanks to Snowflake AISQL to explain journeys, attribution models,  association rules and derive insights (summarize and analyze results, describe behaviors and even suggest actions !)

Identifying, visualizing and explaining paths can itself be actionable and often uncovers an area of interest for additional analysis. First, the picture revealed by path analysis can be further enriched with attribution and association analysis. Attribution helps quantify the contribution of individual touchpoint to a defined outcome, while association analysis uncovers relationships between events that frequently occur together. Together, these techniques provide a holistic understanding of event sequences, enabling data-driven decision-making and uncovering new opportunities for customer experience optimization. Second, path insights can be used directly to predict outcomes or to derive behavioral features (such as the frequency of specific patterns). These features can then be integrated into existing predictive models, enhancing their accuracy and enabling deeper behavioral segmentation. 

## Setup

To set up Snowpath in your own Snowflake account, follow these steps:

1. Download this repository to your local machine.
2. Create a new database in Snowflake for the sample data and application
```sql
create or replace database snowpath;
create or replace schema app;
create or replace stage app_stage;
```
3. Upload all *.py, *.yml, and *.png files to the created `app_stage`.
4. Create the streamlit app with the following command.
```sql
CREATE STREAMLIT snowpath_streamlit
  FROM @snowpath.app.app_stage
  MAIN_FILE = 'streamlit_app.py'
  QUERY_WAREHOUSE = default_wh;
```
5. TODO - Create synthetic data for the app
