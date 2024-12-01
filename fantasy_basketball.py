from itertools import chain
import math
import pandas
import numpy as np
from gekko import GEKKO


TEAM_ERIC = ["Shai Gilgeous-Alexander", "Zach LaVine", "LeBron James", "Kevin Durant", "Daniel Gafford", "Jalen Brunson", "Lauri Markkanen", "Zion Williamson", "Franz Wagner", "Jalen Duren", "Kawhi Leonard", "Tyler Herro", "Josh Giddey"]
TEAM_PAUL = ["Cade Cunningham", "Derrick White", "Jaylen Brown", "Karl-Anthony Towns", "Anthony Davis", "Damian Lillard", "Jalen Williams", "Brook Lopez", "OG Anunoby", "Chris Paul", "Austin Reaves", "Kristaps Porziņģis", "Alex Caruso"]
TEAM_HENRY = ["Trae Young", "Anthony Edwards", "Brandon Miller", "Jabari Smith Jr.", "Rudy Gobert", "Ja Morant", "P.J. Washington", "UTIL", "James Harden", "Dejounte Murray", "Donte DiVincenzo", "Nic Claxton", "Jarrett Allen", "Cam Thomas"]
TEAM_KAI = ["Luka Dončić", "Jimmy Butler", "Scottie Barnes", "Pascal Siakam", "Domantas Sabonis", "Coby White", "DeMar DeRozan", "Bam Adebayo", "Alperen Sengun", "Nikola Vučević", "Jonathan Kuminga", "Josh HartJ", "Stephon Castle"]
TEAM_BILL = ["LaMelo Ball", "Kyrie Irving", "Kyle Kuzma", "Keegan Murray", "Victor Wembanyama", "Jamal Murray", "Paul George", "Fred VanVleet", "Immanuel Quickley", "Walker Kessler", "Joel Embiid", "Devin Vassell", "Tobias Harris"]
TEAM_JAMES = ["De'Aaron Fox", "Donovan Mitchell", "Miles Bridges", "Giannis Antetokounmpo", "Jaren Jackson Jr.", "Jalen Green", "Buddy Hield", "Myles Turner", "D'Angelo Russell", "Anfernee Simons", "Jusuf Nurkić", "Amen Thompson", "Grayson Allen"]
TEAM_ELIAS = ["Stephen Curry", "Desmond Bane", "Bogdan Bogdanović", "Jayson Tatum", "Nikola Jokić", "Tyrese Maxey", "Julius Randle", "Jalen Johnson", "CJ McCollum", "Darius Garland", "Zach Edey", "Brandon Ingram", "Draymond Green"]

FORBIDDEN_PLAYERS = set(chain.from_iterable([
  TEAM_ERIC,
  TEAM_PAUL,
  TEAM_HENRY,
  TEAM_KAI,
  TEAM_BILL,
  TEAM_JAMES,
  TEAM_ELIAS
]))

STAT_CATEGORIES = ["FG%", "FT%", "3P", "TRB", "AST", "STL", "BLK", "TOV", "PTS"]
MIN_NUM_MINUTES_PLAYED = 15

player_stats = pandas.read_csv('2023-2024-player-per-game-stats.csv')

# Remove scrubs who didn't play enough minutes
player_stats = player_stats[player_stats["MP"] >= MIN_NUM_MINUTES_PLAYED].reset_index(drop=True)
print("Keeping a total of {} players".format(len(player_stats)))

# Reduce to only the fields needed for Fantasy Basketball
player_stats = player_stats[["Player", "Pos"] + STAT_CATEGORIES]

# Normalize columns to have MEAN = 0 and STDDEV = 1
player_stats[STAT_CATEGORIES] = player_stats[STAT_CATEGORIES].apply(lambda x: (x-x.mean()) / x.std(), axis=0)
player_stats["TOV"] *= -1 # Invert turnovers, since fewer is better.
print(player_stats)


model = GEKKO()
players = [model.Var(integer=True, lb=0, ub=1) for i in range(len(player_stats))]
stats = {stat_category : model.Var(integer=True, lb=0, ub=1) for stat_category in STAT_CATEGORIES}

total_objective = []
for player_index in range(len(player_stats)):
  player_objective = []
  for stat_category in STAT_CATEGORIES:
    player_stat_value = player_stats[stat_category][player_index]
    player_stat_value = player_stat_value if not math.isnan(x) else -1000
    player_objective += [model.Intermediate(players[player_index] * stats[stat_category] * player_stat_value)]
  
  total_objective += [model.Intermediate(sum(player_objective))]

model.Maximize(sum(total_objective))


# Add constraints to the model
model.Equation(sum(players) <= 10) # Maximum number of chosen players
model.Equation(sum(stats.values()) >= 6) # Minimum number of chosen stats to optimize for

num_pg, num_sg, num_sf, num_pf, num_c = 0, 0, 0, 0, 0
num_g, num_f = 0, 0

for player_index in range(len(player_stats)):
  position = player_stats["Pos"][player_index]
  match position:
    case "PG":
      num_pg += players[player_index]
      num_g += players[player_index]
    case "SG":
      num_sg += players[player_index]
      num_g += players[player_index]
    case "SF":
      num_sf += players[player_index]
      num_f += players[player_index]
    case "PF":
      num_pf += players[player_index]
      num_f += players[player_index]
    case "C":
      num_c += players[player_index]      

model.Equation(num_pg > 1)
model.Equation(num_sg > 1)
model.Equation(num_sf > 1)
model.Equation(num_pf > 1)

model.Equation(num_g > 3)
model.Equation(num_f > 3)
model.Equation(num_c > 3)

for player_index in range(len(player_stats)):
  player = player_stats["Player"][player_index]
  if player in FORBIDDEN_PLAYERS:
    model.Equation(players[player_index] == 0)


model.options.SOLVER = 1 # APOPT solver
model.solve()

selected_stat_categories = {stat_category for stat_category, value in stats.items() if value[0] > 0}
print("\nSelected Categories: {}".format(selected_stat_categories))

print("\nSelected Players:")
selected_player_stats = player_stats[[player[0] > 0 for player in players]]
print(selected_player_stats)

print("\nExpected Score Per Category:")
print(selected_player_stats[list(selected_stat_categories)].sum(axis=0).sort_values(axis=0, ascending=False))


