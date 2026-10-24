from kagglehub import KaggleDatasetAdapter
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import datetime
from IPython.display import display
import plotly.express as px

# Have to load the environment variable prior to loading the kagglehub and kaggle libraries in python
load_dotenv()
import kagglehub
import kaggle
kaggle.api.authenticate()

 # Load appearances DataFrame
acctStr = "davidcariboo/player-scores"
dfStr_appearances = "appearances.csv"
app_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    acctStr,
    dfStr_appearances
)

# Load clubs df
dfStr_clubs = "clubs.csv"
club_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    acctStr,
    dfStr_clubs
)

# Load competitions df
dfStr_comps = "competitions.csv"
comps_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    acctStr,
    dfStr_comps
)

# Load players df
dfStr_players = "players.csv"
players_df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    acctStr,
    dfStr_players
)

YEAR_BEFORE = 2024
CURR_YEAR = 2025
app_df["date"] = pd.to_datetime(app_df["date"])
app_df_filt = app_df[(app_df["date"] > f"{YEAR_BEFORE}-12-31") & (app_df["date"] <= f"{CURR_YEAR}-12-31")]

# merging on the players club id in the appearances df
df_merge = app_df_filt.merge(club_df, how="inner", left_on="player_current_club_id", right_on="club_id")

# grabbing player data from 2025 or greater
player_filt = players_df[(players_df["last_season"] > YEAR_BEFORE) & (players_df["last_season"] <= CURR_YEAR)]

# merging the player data with the merged df
df_merge = df_merge.merge(player_filt, how="inner", on="player_id")
# print(df_merge.columns)

# grabbing the columns and rows that are pertinent to the league that we are interested in
LEAGUE_CODE = "GB1"
COLUMNS = [
    'first_name', 
    'last_name',
    'player_name',
    'goals', 
    'assists',
    'yellow_cards',
    'red_cards',
    'minutes_played', 
    'domestic_competition_id', 
    'sub_position', 
    'position', 
    'foot',
    'current_club_name',
    'market_value_in_eur', 
    'highest_market_value_in_eur',
    'country_of_citizenship'
]
PREM_TEAMS = [
	"Arsenal",
	"Aston Villa",
	"Bournemouth",
	"Brentford",
	"Brighton & Hove Albion",
	"Burnley",
	"Chelsea",
	"Crystal Palace",
	"Everton",
	"Fulham",
	"Leeds United",
	"Liverpool",
	"Manchester City",
	"Manchester United",
	"Newcastle United",
	"Nottingham Forest",
	"Sunderland",
	"Tottenham",
	"West Ham United",
	"Wolverhampton"
]

df_merge = df_merge.loc[:, COLUMNS]

df_merge_eng = df_merge[df_merge["domestic_competition_id"] == LEAGUE_CODE]

pattern = "|".join(PREM_TEAMS)
df_merge_eng = df_merge_eng[df_merge_eng["current_club_name"].str.contains(pattern, case=False, regex=True)]

# checking things
# print(df_merge_eng["position"].unique())
# print(df_merge_eng.isna().sum())

# calc the goals and assists per 90 minutes for attackers in the prem
df_merge_eng_att = df_merge_eng[df_merge_eng["position"] == "Attack"]
df_merge_eng_att["goals_per_pt"] = df_merge_eng_att["goals"] / (df_merge_eng_att["minutes_played"]/90)
df_merge_eng_att["assists_per_pt"] = df_merge_eng_att["assists"] / (df_merge_eng_att["minutes_played"]/90)
df_merge_eng_att["ga_per_pt"] = df_merge_eng_att["goals_per_pt"] + df_merge_eng_att["assists_per_pt"]
df_merge_eng_att = df_merge_eng_att.dropna(subset=['market_value_in_eur', 'highest_market_value_in_eur'])

# create another dataframe that averages every observation for each unique player in the dataframe, should have 172 observations
print(df_merge_eng_att["player_name"].nunique())
df_att_eng_plot = df_merge_eng_att.groupby(["player_name"])[["goals_per_pt", "assists_per_pt", "ga_per_pt", "minutes_played", "market_value_in_eur", "yellow_cards", "red_cards"]].mean()

# filter out players that have less time than 45 minutes on average
df_att_eng_plot = df_att_eng_plot[df_att_eng_plot["minutes_played"] >= 45]
df_att_eng_plot = df_att_eng_plot.reset_index(drop=False, names="player_name")
# merging to get the country each player is from
df_att_eng_plot = df_merge_eng_att[["player_name", "country_of_citizenship", "current_club_name", "foot"]].merge(df_att_eng_plot, how="inner", on="player_name")
# creating a color column based on average time played
def color_mp(x):
    if 45 <= x < 60:
        return "45-60 Minutes"
    elif 60 <= x < 75:
        return "60-75 Minutes"
    elif 75 <= x <= 90:
        return ">75 Minutes"
    else:
        return "Substitute Minutes"
df_att_eng_plot["mp_color"] = df_att_eng_plot["minutes_played"].apply(lambda x: color_mp(x))

# dropping duplicate columns
df_att_eng_plot = df_att_eng_plot.drop_duplicates()

# saving the csv
print(len(df_att_eng_plot))
df_att_eng_plot.to_csv("english_attackers.csv", index=False)

# plot the hoverable scatterplot
fig = px.scatter(
    df_att_eng_plot,
    x="goals_per_pt",
    y="assists_per_pt",
    size="market_value_in_eur",
    color="mp_color",
    category_orders={
        "mp_color": ["45-60 Minutes", "60-75 Minutes", ">75 Minutes"]
    },
    hover_name="player_name",  # Appears in bold at the top of the hover box
    hover_data=[
        "market_value_in_eur",
        "current_club_name"
    ],  # Additional items to list in the hover box
)
fig.update_layout(title={"text": f"{YEAR_BEFORE+1}/{CURR_YEAR+1} Striker Efficiency", "x": 0.5, "xanchor": "center"})
fig.update_layout(legend_title_text="Average Playing Time")
fig.show()