import pyomo.environ as pyo
import pandas as pd

# Defining each of the sets that will be useful for all of the DMUs that we will be comparing
att_df = pd.read_csv("./english_attackers.csv")

inputs = ["red_cards", "yellow_cards", "minutes_played", "market_value_in_eur"]
outputs = ["goals_per_pt", "assists_per_pt"]
att_df = att_df.dropna(subset=inputs + outputs).reset_index(drop=True)

# Build the abstract model
model = pyo.AbstractModel()
model.I = pyo.Set()
model.IN = pyo.Set()
model.OUT = pyo.Set()
model.o = pyo.Param(within=model.I)  

model.x = pyo.Param(model.IN, model.I)  
model.y = pyo.Param(model.OUT, model.I)  

model.lamb = pyo.Var(model.I, bounds=(0, 1))
model.slack = pyo.Var(model.IN, domain=pyo.NonNegativeReals)
model.surplus = pyo.Var(model.OUT, domain=pyo.NonNegativeReals)

def obj_rule(m):
    return -sum(m.slack[k] for k in m.IN) + -sum(m.surplus[r] for r in m.OUT)
model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

def output_rule(m, r):
    return sum(m.y[r, i] * m.lamb[i] for i in m.I) - m.surplus[r] == m.y[r, m.o]
model.output_cons = pyo.Constraint(model.OUT, rule=output_rule)

def input_rule(m, k):
    return sum(-m.x[k, i] * m.lamb[i] for i in m.I) - m.slack[k] == -m.x[k, m.o]
model.input_cons = pyo.Constraint(model.IN, rule=input_rule)

def convexity_rule(m):
    return sum(m.lamb[i] for i in m.I) == 1
model.convexity = pyo.Constraint(rule=convexity_rule)

# Put the data in the dictionary format Pyomo expects
players = att_df.index.tolist()
base_data = {
    "I":   {None: players},
    "IN":  {None: inputs},
    "OUT": {None: outputs},
    "x":   {(k, i): att_df.loc[i, k] for k in inputs for i in players},
    "y":   {(r, i): att_df.loc[i, r] for r in outputs for i in players},
}

# Create an instance per player and solve that model, also save model variables to df
results_data = []
solver = pyo.SolverFactory("highs")
for o in players:
    data = {None: {**base_data, "o": {None: o}}}
    instance = model.create_instance(data)
    result = solver.solve(instance)
    results_data.append({
        "name": att_df.loc[o, "player_name"], 
        "obj_value": pyo.value(instance.obj),
        "reds_slack": pyo.value(instance.slack["red_cards"]),
        "yellows_slack": pyo.value(instance.slack["yellow_cards"]),
        "mins_slack": pyo.value(instance.slack["minutes_played"]),
        "price_slack": pyo.value(instance.slack["market_value_in_eur"]),
        "gpg_surplus": pyo.value(instance.surplus["goals_per_pt"]),
        "apg_surplus": pyo.value(instance.surplus["assists_per_pt"])
    })
results_df = pd.DataFrame(results_data)
results_df.to_csv("./dea_additive_model_results.csv", index=False)
print("Done.")