import pandas as pd
eng_att_df = pd.read_csv("./english_attackers.csv")
dea_results_df = pd.read_csv("./dea_additive_model_results.csv")

dea_results_labs = dea_results_df.merge(eng_att_df, how="inner", left_on="name", right_on="player_name")
dea_results_labs = dea_results_labs.drop(columns=["player_name"])

# find the optimal players
optimal_df = dea_results_labs[abs(dea_results_labs["obj_value"]) <= 1]
print("The players that were optimal are listed below: ")
with pd.option_context('display.max_columns', None):
    print(optimal_df[["name", "current_club_name", "foot", "mp_color", "market_value_in_eur", "goals_per_pt", "assists_per_pt"]])

# suboptimal players
suboptimal_df = dea_results_labs[abs(dea_results_labs["obj_value"]) > 1]
print("The suboptimal players are displayed below: ")
with pd.option_context('display.max_columns', None):
    print(suboptimal_df)