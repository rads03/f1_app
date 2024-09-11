#!/usr/bin/env python
# coding: utf-8

# In[2]:


import fastf1 as f1
from fastf1 import plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import plotly.graph_objects as go
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter


# In[3]:


def get_race_dfs(year, location):
    race = f1.get_session(year, location, 'R')
    race.load(weather=True)
    df=race.laps
    df =  df[df['Deleted']==False]
    df = df.sort_values(by=['LapNumber','Position'], ascending=[False, True]).reset_index(drop=True)

    df.LapTime = df.LapTime.fillna(df['Sector1Time']+df['Sector2Time']+df['Sector3Time'])
    df.LapTime = df.LapTime.dt.total_seconds()
    df.Sector1Time = df.Sector1Time.dt.total_seconds()
    df.Sector2Time = df.Sector2Time.dt.total_seconds()
    df.Sector3Time = df.Sector3Time.dt.total_seconds()
    df['LapNumber'] = df['LapNumber'].astype(int)
    df['Stint'] = df['Stint'].astype(int)
    
    df_weather = race.weather_data.copy()
    df_weather['Time'] = df_weather['Time'].dt.total_seconds()/60
    df_weather = df_weather.rename(columns={'Time':'SessionTime(Minutes)'})

    rain=df_weather.Rainfall.eq(True).any()
    
    return race, df, df_weather


# In[6]:


def get_quali_dfs(year, location):
    quali = f1.get_session(year, location, 'Q')
    quali.load(weather=True)
    dfq=quali.laps
    dfq['LapTime'] = dfq['LapTime'].dt.total_seconds()
    dfq['Sector1Time'] = dfq['Sector1Time'].dt.total_seconds()
    dfq['Sector2Time'] = dfq['Sector2Time'].dt.total_seconds()
    dfq['Sector3Time'] = dfq['Sector3Time'].dt.total_seconds()
    q1, q2, q3 = dfq.split_qualifying_sessions()
    
    return q1, q2, q3


# In[8]:


def highlight_last_five_rows(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    styles.iloc[-5:] = 'background-color: red'
    return styles


# In[9]:


def get_quali_results(q1, q2, q3):

    rows_to_append = []
    for driver in q1['Driver'].unique():
        rows_to_append.append(q1.loc[q1[q1['Driver'] == driver].LapTime.idxmin()])
    q1_pos = pd.DataFrame(rows_to_append)
    q1_pos['Position'] = q1_pos['LapTime'].rank(method='min', ascending=True).astype(int)
    q1_pos = q1_pos.sort_values(by='Position').reset_index(drop=True)
    q1_pos = q1_pos[['Driver', 'LapTime', 'Position', 'FreshTyre', 'TyreLife', 'Compound']]
    q1_pos.reset_index(drop=True, inplace=True)

    rows_to_append = []
    for driver in q2['Driver'].unique():
        rows_to_append.append(q2.loc[q2[q2['Driver'] == driver].LapTime.idxmin()])
    q2_pos = pd.DataFrame(rows_to_append)
    q2_pos['Position'] = q2_pos['LapTime'].rank(method='min', ascending=True).astype(int)
    q2_pos = q2_pos.sort_values(by='Position').reset_index(drop=True)
    q2_pos = q2_pos[['Driver', 'LapTime', 'Position', 'FreshTyre', 'TyreLife', 'Compound']]

    rows_to_append = []
    for driver in q3['Driver'].unique():
        rows_to_append.append(q3.loc[q3[q3['Driver'] == driver].LapTime.idxmin()])
    q3_pos = pd.DataFrame(rows_to_append)
    q3_pos['Position'] = q3_pos['LapTime'].rank(method='min', ascending=True).astype(int)
    q3_pos = q3_pos.sort_values(by='Position').reset_index(drop=True)
    q3_pos = q3_pos[['Driver', 'LapTime', 'Position', 'FreshTyre', 'TyreLife', 'Compound']]
    q3_pos.reset_index(drop=True, inplace=True)

    return q1_pos, q2_pos, q3_pos


# In[11]:


def get_gap_to_pole(q3_pos):
    
    driver_names = q3_pos['Driver'].unique()
    pole_time = q3_pos['LapTime'].min()
    q3_pos['GapToPole'] = q3_pos['LapTime'] - pole_time

    fig, ax = plt.subplots(figsize=(14, 16))
    
    custom_colors = ['#ffa500', '#ff9300', '#ff8000', '#ff6e00', '#ff5c00', '#ff4900', '#ff3700', '#ff2500', '#ff1200', '#ff0000'] 
    custom_palette = sns.color_palette(custom_colors)

    sns.barplot(x='GapToPole', y='Driver', data=q3_pos, palette=custom_palette, ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    plt.grid(False)
    
    ax.set_yticks(range(len(driver_names)))
    ax.set_yticklabels(driver_names, fontsize=25, fontweight='bold')
    
    fig.patch.set_facecolor('#222222')
    ax.set_facecolor('#222222')
    ax.tick_params(colors='white', which='both')

    for index, value in enumerate(q3_pos['GapToPole']):
        ax.text(value, index, f"{value:.3f}s", va='center', ha='left', color='white', fontweight='bold', fontsize=25)

    return fig


# In[13]:


def get_sector_times(q3):
    sector_times = q3.groupby('Driver')[['Sector1Time', 'Sector2Time', 'Sector3Time']].mean()
    sector_times.reset_index(inplace=True)
    sector_times_melted = sector_times.melt(id_vars='Driver', var_name='Sector', value_name='Time')

    fig, ax = plt.subplots(figsize=(14, 8))
    custom_colors = ['#B0C4DE', '#1f77b4', 'orange']

    sns.barplot(x='Driver', y='Time', hue='Sector', data=sector_times_melted, palette=custom_colors, ax=ax)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend()
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.tick_params(colors='white', which='both')


    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    return fig


# In[15]:


def get_circuit_map(df, driver=None):
    if driver is None:
        unique_driver_numbers = df.index.unique()
        if len(unique_driver_numbers) == 0:
            raise ValueError("No drivers found in the dataset.")
        driver_number = unique_driver_numbers[0]
    else:
        if driver not in df.index:
            raise ValueError(f"Driver {driver} did not participate in this race.")
    
    telemetry = df.loc[driver_number]
    x_pos = telemetry['X'].values
    y_pos = telemetry['Y'].values
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    num_glow_layers = 10 
    for i in range(1, num_glow_layers + 1):
        ax.plot(x_pos, y_pos, color='orange', linewidth=12, alpha=0.5) 
    
    ax.plot(x_pos, y_pos, color='white', linewidth=5)  
    ax.set_facecolor('black')

    fig.patch.set_facecolor('black')
    
    return fig



# In[53]:


def compare_driver_stats(df, driver_1, driver_2, stat, year, driver_mappings):
 
    driver_mapping_dict = driver_mappings.get(year)
    
    if not driver_mapping_dict:
        raise ValueError(f"No driver mapping dictionary found for the year {year}")
    
    driver1 = driver_mapping_dict.get(driver_1)
    driver2 = driver_mapping_dict.get(driver_2)
    
    if driver1 is None or driver2 is None:
        raise ValueError(f"Driver names '{driver_1}' or '{driver_2}' not found in the mapping dictionary for the year {year}")
    
    telemetry1 = df.loc[driver1]
    telemetry2 = df.loc[driver2]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(x=telemetry1.index, y=telemetry1[stat], label=driver_1, linewidth=3.5, color='cornflowerblue')
    sns.lineplot(x=telemetry2.index, y=telemetry2[stat], label=driver_2, linewidth=3.5, color='orange')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    legend = ax.legend(fontsize='20', loc='best', facecolor='black', edgecolor='white', labelcolor='white')
    
    for text in legend.get_texts():
        text.set_color('white')
        text.set_fontweight('bold')
    
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.tick_params(colors='white', which='both', labelsize=15)
    ax.set_xticklabels([])

    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(True, linestyle='--', color='white', alpha=0.7)
    plt.tight_layout()
    plt.close()
    
    return fig
    


# In[55]:


def compare_teammates(df, Driver1, Driver2):
    
    def format_lap_time(seconds):
        minutes, sec = divmod(seconds, 60)
        return f'{int(minutes):02}:{int(sec):02}'
    
    colors = {Driver1: 'cornflowerblue', Driver2: 'orange'}
    df_sorted = df.sort_values(by='LapNumber')
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for Driver in [Driver1, Driver2]:
        driver_data = df_sorted[df_sorted['Driver'] == Driver]
        sns.lineplot(x='LapNumber', y='LapTime', data=driver_data, marker='o', markersize=4, 
                     markerfacecolor='white', markeredgewidth=4, linestyle='-', linewidth=3.5, 
                     color=colors.get(Driver, 'gray'), label=Driver, ax=ax)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    legend = ax.legend(loc='best', fontsize='20', facecolor='black', edgecolor='white', labelcolor='white')
    
    for text in legend.get_texts():
        text.set_color('white')
        text.set_fontweight('bold')
    
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.tick_params(colors='white', which='both', labelsize=15)

    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_lap_time(x)))
    ax.grid(True, linestyle='--', color='white', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


# In[21]:


def get_stint_laptime_trend(df):
    stint_order = sorted(df['Stint'].unique())

    for stint in stint_order:
        dfs = df[df['Stint'] == stint]

        dfs = dfs.groupby('Driver').apply(lambda x: x[:-1]).reset_index(drop=True)

        last_lap_idx = dfs.groupby('Driver').apply(lambda x: x.tail(1).index).explode()
        dfs = dfs.drop(last_lap_idx)

        model = LinearRegression()
        X = dfs[['LapNumber']].values 
        y = dfs['LapTime'].values
        model.fit(X, y)

        dfs['PredictedLapTime'] = model.predict(X)
        dfs['DetrendedLapTime'] = dfs['LapTime'] - dfs['PredictedLapTime']

        g = sns.FacetGrid(dfs, col="Driver", col_wrap=3, height=4, aspect=1.5, palette=Color_map)
        g.map_dataframe(sns.scatterplot, x="LapNumber", y="LapTime", hue="Driver", palette=Color_map, legend=False)
        g.map_dataframe(sns.lineplot, x="LapNumber", y="PredictedLapTime", color="red", linestyle="--")
        g.set_axis_labels("Lap Number", "Lap Time (seconds)")
        g.set_titles("{col_name}")

        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f'LapTime vs LapNumber with Trend Line for Each Driver in Stint {stint}')
        plt.show()


# In[22]:


def get_wind_map(df_weather):

    df_weather_copy = df_weather.copy()
    df_weather_copy['WindDirection'] = np.deg2rad(df_weather_copy['WindDirection'])

    norm = mcolors.Normalize(vmin=df_weather_copy['WindSpeed'].min(), vmax=df_weather_copy['WindSpeed'].max())
    cmap = cm.get_cmap('Blues') 

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

    for index, row in df_weather_copy.iterrows():
        color = cmap(norm(row['WindSpeed']))
        ax.scatter(row['WindDirection'], row['WindSpeed'], s=100, color=color, alpha=0.75)

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.set_rlabel_position(90)
    ax.set_ylim(0, df_weather['WindSpeed'].max() + 5)
    ax.tick_params(colors='white', which='both', labelsize=15)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.035, pad=0.05)
    cbar.set_label('Wind Speed (m/s)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    cbar.ax.set_yticklabels([str(label) for label in cbar.ax.get_yticks()], color='white')

    return fig


# In[24]:


def get_temp_graph():
    plt.figure(figsize=(16, 10))
    plt.plot(df_weather['SessionTime(Minutes)'], df_weather['AirTemp'], marker='o', linestyle='-', label='Air Temperature')
    plt.plot(df_weather['SessionTime(Minutes)'], df_weather['TrackTemp'], marker='o', linestyle='-', label='Track Temperature')
    plt.title('Track Temperature & Air Temperature (in Celcius)')
    plt.xlabel('Session Time(Minutes)')
    plt.ylabel('Temperature')
    plt.grid(True)
    plt.legend()
    plt.show()


# In[25]:


def get_humidity_graph():
    plt.figure(figsize=(16, 10))
    plt.plot(df_weather['SessionTime(Minutes)'], df_weather['Humidity'], marker='o', linestyle='-')
    plt.title('Track Humidity (%)')
    plt.xlabel('Session Time(Minutes)')
    plt.ylabel('Humidity')
    plt.grid(True)
    plt.show()


# In[26]:


def get_pressure_graph():
    plt.figure(figsize=(16, 10))
    plt.plot(df_weather['SessionTime(Minutes)'], df_weather['Pressure'], marker='o', linestyle='-')
    plt.title('Air Pressure (mbar)')
    plt.xlabel('Session Time(Minutes)')
    plt.ylabel('Pressure')
    plt.grid(True)
    plt.show()


# In[27]:


def get_rainfall_graph():
    plt.step(df_weather['SessionTime(Minutes)'], df_weather['Rainfall'], where='post', color='blue', label='Rainfall')
    plt.title('Rainfall')
    plt.xlabel('Session Time(Minutes)')
    plt.ylabel('Rainfall (True/False)')
    plt.yticks([0, 1], ['No Rain', 'Rain'])
    plt.show()


# In[28]:


def get_circuit_corners_map():
    circuit_info = race.get_circuit_info()
    corners_df = circuit_info.corners
    marshal_lights_df = circuit_info.marshal_lights
    marshal_sectors_df = circuit_info.marshal_sectors
    
    first_corner = corners_df.iloc[0]
    last_corner = corners_df.iloc[-1]

    plt.figure(figsize=(10, 8))
    plt.plot(corners_df['X'], corners_df['Y'], 'bo-', label='Corners')

    plt.plot([last_corner['X'], first_corner['X']], [last_corner['Y'], first_corner['Y']], 'b-')

    plt.title('Track Corners with Manual Connection')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()


# In[69]:


def plot_track_dominance(df, driver1, driver2):
    
    telemetry1 = df.iloc[[df[df['Driver']==driver1].LapTime.idxmin()]].get_telemetry().add_distance()
    telemetry2 = df.iloc[[df[df['Driver']==driver2].LapTime.idxmin()]].get_telemetry().add_distance()

    x_pos1 = telemetry1['X'].values
    y_pos1 = telemetry1['Y'].values
    x_pos2 = telemetry2['X'].values
    y_pos2 = telemetry2['Y'].values
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def plot_dominance_along_track(x1, y1, x2, y2, color1, color2):
        min_length = min(len(x1), len(x2)) - 1
        for i in range(min_length):
     
            speed1 = telemetry1['Speed'].values[i]
            speed2 = telemetry2['Speed'].values[i]
            
            if speed1 > speed2:
                dominance_color = color1
            else:
                dominance_color = color2
            
            ax.plot([x1[i], x1[i+1]], [y1[i], y1[i+1]], color=dominance_color, linewidth=10)
            ax.plot([x2[i], x2[i+1]], [y2[i], y2[i+1]], color=dominance_color, linewidth=10)
    
    plot_dominance_along_track(x_pos1, y_pos1, x_pos2, y_pos2, 'orange', 'lightsteelblue')

    indigo_patch = plt.Line2D([0], [0], color='orange', linewidth=8, alpha=0.6)
    blue_patch = plt.Line2D([0], [0], color='lightsteelblue', linewidth=8, alpha=0.6)
    
    legend = ax.legend([indigo_patch, blue_patch], 
          [f'{driver1} Dominance', f'{driver2} Dominance'], 
          loc='best', 
          fontsize=20, 
          facecolor='black', 
          edgecolor='white',
          labelcolor='white')
    
    for text in legend.get_texts():
        text.set_color('white')
        text.set_fontweight('bold')


    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    plt.tight_layout()
    
    return fig


# In[59]:


def create_race_results_table(results):
    def format_timedelta(td, is_first_row, Status=False):
        if pd.isna(td):
            return Status
        total_seconds = td.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        
        if is_first_row:
            return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
        else:
            return f"+ {total_seconds:.3f} seconds"

    # Ensure the 'Time' column is in timedelta format
    if results['Time'].dtype == 'object':
        # Try to convert the 'Time' column to timedelta
        results['Time'] = pd.to_timedelta(results['Time'], errors='coerce')

    results['Formatted_Time'] = results.apply(
        lambda row: format_timedelta(row['Time'], False, row['Status']), axis=1
    )

    if not results.empty:
        results.iloc[0, results.columns.get_loc('Formatted_Time')] = format_timedelta(results.iloc[0]['Time'], True, results.iloc[0]['Status'])

    df_results = results[['BroadcastName', 'TeamName', 'ClassifiedPosition', 'Formatted_Time']]
    df_results.columns = ['BroadcastName', 'TeamName', 'ClassifiedPosition', 'Formatted_Time']


    headerColor = '#333333'  # Dark grey for header
    rowEvenColor = '#444444'  # Slightly lighter grey for even rows
    rowOddColor = '#555555'   # Slightly lighter grey for odd rows
    winnerColor = 'orange'   # Bright color for the winner row

    column_width = [0.15, 0.15, 0.05, 0.1]

    fill_colors = [winnerColor if i == 0 else rowOddColor if i % 2 == 0 else rowEvenColor for i in range(len(df_results))]
    font_sizes = [16 if i == 0 else 12 for i in range(len(df_results))]
    font_weights = ['bold' if i == 0 else 'normal' for i in range(len(df_results))]

    fig = go.Figure(data=[go.Table(
        columnwidth=column_width,
        header=dict(
            values=['<b>Driver</b>', '<b>Team</b>', '<b>Pos</b>', '<b>Time</b>'],
            fill_color=headerColor,
            align='center',
            font=dict(size=16, color='white', family='Arial'),
            height=30,
            line_color='rgba(0,0,0,0)', 
            line_width=0
        ),
        cells=dict(
            values=[df_results['BroadcastName'], 
                    df_results['TeamName'], df_results['ClassifiedPosition'], 
                    df_results['Formatted_Time']],
            fill_color=[fill_colors],
            align='center',
            font=dict(size=12, color='white', family='Arial'), 
            height=30,
            line_color='rgba(0,0,0,0)',
            line_width=0
        )
    )])

    fig.update_layout(
        paper_bgcolor='black',  # Set background color of the entire figure
        margin=dict(l=10, r=10, b=10, t=40),
    )
    return fig


# In[61]:


def plot_tyre_strategy(df):
    
    colors = {
    'MEDIUM': '#f6d53b',  
    'SOFT': '#e31c1f',   
    'HARD': '#Fefefe',   
    'INTERMEDIATE': '#1faa4a',
    'WET': '#1171bd'}

    driver_list = df['Driver'].unique()
    
    grouped_df = df.groupby(['Driver', 'Stint', 'Compound']).agg({'LapNumber': 'count'}).reset_index()
    grouped_df.rename(columns={'LapNumber': 'Number of Laps'}, inplace=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    gap = 0.2
    radius = 0.3

    driver_names = sorted(set(driver_list))
    driver_mapping = {name: i for i, name in enumerate(driver_names)}

    driver_positions = {driver: 0 for driver in driver_names}

    for i, row in grouped_df.iterrows():
        driver = row['Driver']
        stint = row['Stint']
        compound = row['Compound']
        num_laps = row['Number of Laps']
        
        left_position = driver_positions[driver]
        
        height = 0.7
        width = num_laps 
        left = left_position
        y_position = driver_mapping[driver]
  
        
        rect = FancyBboxPatch((left, y_position - height / 2), width, height,
                              boxstyle="round,pad=0.05,rounding_size={}".format(radius),
                              edgecolor=None, linewidth=0.5, facecolor=colors[compound],
                              zorder=3)
        ax.add_patch(rect)
        
        ax.text(left + (width / 2), y_position, str(num_laps), 
                va='center', ha='center', fontsize=7, fontweight='bold', color='black')
        
        driver_positions[driver] += num_laps + gap

    ax.set_xlim(0, df['LapNumber'].max() + 1)
    ax.set_ylim(-0.6, len(driver_names)) 
    ax.set_yticks(range(len(driver_names)))
    ax.set_yticklabels(driver_names, fontsize=5)
    ax.tick_params(axis='x', labelsize=5, colors='white')
    ax.tick_params(axis='y', labelsize=5, colors='white')
    plt.grid(False)
    
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
                   
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    
    for tick in ax.get_xticks():
        ax.axvline(x=tick, color='white', linestyle='--', linewidth=0.7, ymax=0.05)

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('white')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('white')
    ax.spines['bottom'].set_linewidth(0.5)

    return fig


# In[ ]:




