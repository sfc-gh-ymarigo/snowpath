# Sequent

Sequent™ native application allows users to easily and visually perform and deep dive into Path Analysis, Attribution Analysis, Association Analysis, Pattern Mining, Behavioral Segmentation and Predictive Modeling by simply specifying a few parameters in drop-down menus. Leveraging advanced techniques, Sequent™ intuitively and visually helps identify touchpoints influencing customer (or machine) behaviours, targets them to create segments, performs cross-population behavioural comparisons, computes rule-based and ML-driven attribution models to understand the contribution of each event preceding a specific outcome, conducts association analysis to uncover hidden patterns and relationships between events, discovers frequent sequential patterns and behavioral signatures through advanced pattern mining, and enables sophisticated behavioral segmentation to group customers based on their journey patterns and characteristics. Sequent™ also harnesses the interpretive and generative power of LLMs thanks to Snowflake AISQL to explain journeys, attribution models, association rules, pattern insights and derive insights (summarize and analyze results, describe behaviors and even suggest actions !) 

Visualizing and identifying paths can itself be actionable and often uncovers an area of interest for additional analysis. First, the picture revealed by path analysis can be further enriched with attribution analysis, association analysis, pattern mining, and behavioral segmentation. Attribution helps quantify the contribution of individual touchpoints to a defined outcome, association analysis uncovers relationships between events that frequently occur together, pattern mining discovers frequent sequential behaviors and hidden temporal dependencies, and behavioral segmentation groups customers into meaningful clusters based on their journey characteristics and patterns. Together, these techniques provide a comprehensive understanding of event sequences, enabling data-driven decision-making and uncovering new opportunities for optimization. Second, path insights can be used directly to predict outcomes (Predictive Modeling) or to derive behavioral features (such as the frequency of specific patterns and sequence signatures). These features can then be integrated into existing predictive models, enhancing their accuracy and enabling deeper customer understanding through advanced segmentation strategies

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
