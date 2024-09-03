#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import fastf1 as f1
from fastf1 import plotting
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import nbimporter
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageOps
from PIL import ImageChops
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# In[3]:


from f1functions import get_race_dfs
from f1functions import get_quali_dfs
from f1functions import highlight_last_five_rows
from f1functions import get_quali_results
from f1functions import get_gap_to_pole
from f1functions import get_sector_times
from f1functions import get_circuit_map
from f1functions import get_wind_map
from f1functions import plot_track_dominance
from f1functions import compare_teammates
from f1functions import compare_driver_stats
from f1functions import create_race_results_table
from f1functions import plot_tyre_strategy
import warnings
warnings.filterwarnings('ignore')


# In[4]:


st.set_page_config(
    page_title="F1 Analytics Dashboard",
    page_icon="🏎️",
    layout="wide")


# In[5]:


st.markdown(
    """
    <style>
    .main {
        background-color: black;
        color: white;
    }
    .block-container {
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# In[6]:


st.markdown(
    """
    <style>
    h4 {
        color: white;
        background-color: black;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# In[7]:


st.markdown(
    """
    <style>
    .css-1kq5h4w {
        color: white !important;
    }
    .css-1kq5h4w span {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# In[8]:


st.markdown(
    """
    <style>
    /* Change color of small headings (#####) to white */
    h5 {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# In[9]:


st.markdown(
    """
    <style>
    /* Change text color in selectboxes */
    .stSelectbox div[data-baseweb="select"] > div {
        color: white;
    }

    /* Change text color in number inputs */
    .stNumberInput input {
        color: white;
    }

    /* Optional: change background color of the input fields */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #333333;
    }

    .stNumberInput input {
        background-color: #333333;
    }

    /* Remove the white outline from selectboxes */
    .stSelectbox div[data-baseweb="select"] {
        border: none;
    }

    /* Remove border around the selectbox */
    .stSelectbox div[data-baseweb="select"] > div {
        border: none;
    }

    /* Change background color of buttons */
    .stButton button {
        background-color: #333333;
        color: white;
    }

    /* Optional: Change button hover effect */
    .stButton button:hover {
        background-color: #555555;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

css = """
<style>
/* Target the class used for metric values */
.st-emotion-cache-1wivap2.e1i5pmia3 {
    color: white !important;
}

/* Target the class used for metric labels */
.st-emotion-cache-q49buc.e1i5pmia2 {
    color: white !important;
}
</style>
"""

# Apply CSS to the Streamlit app
st.markdown(css, unsafe_allow_html=True)


# In[10]:


driver_images = {
    'M VERSTAPPEN': 'pics/VER.png',
    'L NORRIS': 'pics/NOR.png',
    'L HAMILTON': 'pics/HAM.png',
    'O PIASTRI': 'pics/PIA.png',
    'G RUSSELL': 'pics/RUS.png',
    'S PEREZ': 'pics/PER.png',
    'F ALONSO': 'pics/ALO.png',
    'L STROLL': 'pics/STR.png',
    'C LECLERC': 'pics/LEC.png',
    'C SAINZ': 'pics/SAI.png',
    'D RICCIARDO': 'pics/RIC.png',
    'Y TSUNODA': 'pics/TSU.png',
    'P GASLY': 'pics/GAS.png',
    'E OCON': 'pics/OCO.png',
    'A ALBON': 'pics/ALB.png',
    'L SARGEANT': 'pics/SAR.png',
    'V BOTTAS': 'pics/BOT.png',
    'G ZHOU': 'pics/ZHO.png',
    'K MAGNUSSEN': 'pics/MAG.png',
    'N HULKENBERG': 'pics/HUL.png'
}


# In[11]:


def get_available_locations(year):
    calendar = f1.get_event_schedule(year)
    calendar['EventDate'] = pd.to_datetime(calendar['EventDate'])
    today = datetime.now()
    past_events = calendar[calendar['EventDate'] < today]
    locations = past_events[['Location', 'Country']].drop_duplicates() 
    most_recent_event = calendar[calendar['EventDate'] <= today].sort_values(by='EventDate', ascending=False).iloc[0]
    default_location = most_recent_event['Location']
    default_country = most_recent_event['Country']
    return locations, default_location, default_country, calendar


# In[12]:


col1, col2, col3 = st.columns((2.9, 2.9, 2.8), gap='large')

with col1:
    year = st.number_input('', min_value=2000, max_value=2024, value=2024)

with col2:
    locations, default_location, default_country, calendar = get_available_locations(year)
    location_options = [f"{loc} ({country})" for loc, country in zip(locations['Location'], locations['Country'])]
    default_location_index = location_options.index(f"{default_location} ({default_country})")
    location_country = st.selectbox('', location_options, index=default_location_index)
    location = location_country.split(' (')[0]

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button('Get Race Info'):
        st.session_state.year = year
        st.session_state.location = location
        st.session_state.load_data = True


# In[13]:


col = st.columns((3, 3, 3), gap='large')


# In[14]:


with col[1]:
    if 'load_data' in st.session_state and st.session_state.load_data:
        try:
            race, df, df_weather = get_race_dfs(year, location)
            q1, q2, q3 = get_quali_dfs(year, location)
            q1_pos, q2_pos, q3_pos = get_quali_results(q1, q2, q3)
            results = race.results
            fig1 = get_gap_to_pole(q3_pos)
            
            st.write("\n\n")
            st.write("\n\n")
            st.markdown("#### Gap to Pole")
            st.pyplot(fig1)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")


# In[19]:


with col[0]:
     if 'q3' in locals() and 'q3_pos' in locals() and 'race' in locals() and 'df_weather' in locals():
        try: 
            WinningDriver = results.iloc[0]['BroadcastName']
            WinningDriver = WinningDriver.values[0] if isinstance(WinningDriver, pd.Series) else WinningDriver
            
            driver_image_path = driver_images.get(WinningDriver, 'pics/default_driver.png')
            
            st.write("\n\n")
            st.write("\n\n")
            st.write("\n\n")
            st.image(Image.open(driver_image_path), caption=f"Winning Driver: {WinningDriver}", use_column_width=True)  
            
            
            fig3 = get_circuit_map(df, race, driver=None)
            st.write("\n\n")
            st.pyplot(fig3)
            
            avg_track_temp = df_weather['TrackTemp'].mean()
            avg_air_temp = df_weather['AirTemp'].mean()

            col1, col2 = st.columns([1, 1])

            with col1:
                st.metric(label="Average Track Temperature", value=f"{avg_track_temp:.1f}°C")

            with col2:
                st.metric(label="Average Air Temperature", value=f"{avg_air_temp:.1f}°C")
            
        except Exception as e:
            st.error(f'Error: {e}')


# In[16]:


with col[2]:
    try:
        if 'q3' in locals() and 'q3_pos' in locals() and 'race' in locals() and 'results' in locals():

            fig8 = create_race_results_table(results)
            fig8.update_layout(autosize=True, height=950)

            st.plotly_chart(fig8, use_container_width=True)

    except Exception as e:
        st.error(f'Error: {e}')


# In[17]:


st.write("\n\n")
st.write("\n\n")
if 'df' in locals() and 'race' in locals() and 'results' in locals():
    try:
        plot_type = st.selectbox('Select Plot Type', ['Race Pace Comparison', 'Track Dominance', 'Driver Stat Comparison'])
        
        default_driver1 = results.iloc[0]['Abbreviation']
        default_driver2 = results.iloc[1]['Abbreviation']
        
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                driver1 = st.selectbox('Driver 1', df['Driver'].unique(), index=df['Driver'].unique().tolist().index(default_driver1))
            with col2:
                driver2 = st.selectbox('Driver 2', df['Driver'].unique(), index=df['Driver'].unique().tolist().index(default_driver2))
        
        if plot_type == 'Race Pace Comparison':
            if driver1 and driver2:
                fig6 = compare_teammates(df, driver1, driver2) 
                st.markdown("#### Race Pace Comparison")
                st.pyplot(fig6)
            else:
                st.error('Please select both drivers.')
                
        elif plot_type == 'Track Dominance':
            if driver1 and driver2:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    fig5 = plot_track_dominance(race, driver1, driver2) 
                    st.markdown("#### Track Dominance")
                    st.pyplot(fig5)
            else:
                st.error('Please select both drivers.')
            
        elif plot_type == 'Driver Stat Comparison':
            stat = st.selectbox('Select Stat', ['Speed', 'nGear', 'Throttle', 'DRS'])
                
            if driver1 and driver2 and stat:
                fig7 = compare_driver_stats(df, race, driver1, driver2, stat)
                st.markdown("#### Driver Comparison")
                st.pyplot(fig7)
            else:
                st.error('Please make all selections')
                    
    except Exception as e:
        st.error(f"An error occurred: {e}")


# In[18]:


st.write("\n\n")
st.write("\n\n")
st.write("\n\n")
st.write("\n\n")
if 'df' in locals():
    try:
        fig9 = plot_tyre_strategy(df)
        
        st.image(Image.open('pics/tyres.png'), use_column_width=True)
        st.markdown('##### Tyre Strategy')
        st.pyplot(fig9)
        
    except Exception as e:
        st.error(f'Error: {e}') 


# In[ ]:





# In[ ]:




