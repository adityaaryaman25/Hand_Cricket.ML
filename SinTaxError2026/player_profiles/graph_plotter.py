from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

# 1. Locate the JSON file in the *same* directory as this script
script_dir = Path(__file__).resolve().parent
file_path = script_dir / "26118012.json"

print(f"Loading data from: {file_path}")

# 2. Open and parse the JSON file
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

# 1. Locate the JSON file in the *same* directory as this script
script_dir = Path(__file__).resolve().parent
file_path = script_dir / "26118012.json"

print(f"Loading data from: {file_path}")

# 2. Open and parse the JSON file
with open(file_path, "r") as file:
  data = json.load(file)

# 3. Extract records and load into a pandas DataFrame
records = data.get("records", [])
df = pd.DataFrame(records)

if df.empty:
  print("Error: No records found in the JSON file.")
else:
  print(f"Successfully loaded {len(df)} total records!")

  # 4. Take the last 100 moves for detailed analysis
  last_100 = df.tail(100).copy()

  # 5. Calculate correct predictions (where AI's my_move matches user's move)
  last_100["correct"] = (last_100["my_move"] == last_100["move"]).astype(int)

  # 6. Calculate rolling accuracy over a 10-move window (percentage)
  last_100["rolling_acc"] = (
      last_100["correct"].rolling(window=10, min_periods=1).mean() * 100
  )

  # 7. Generate and display the plot
  plt.figure(figsize=(12, 6))
  plt.plot(
      last_100.reset_index(drop=True).index,
      last_100["rolling_acc"],
      color="teal",
      linewidth=2.5,
      marker="o",
      markersize=4,
  )

  plt.title(
      "AI Progressive Learning Curve: Rolling Prediction Accuracy (Last 100"
      " Moves)",
      fontsize=14,
      fontweight="bold",
  )
  plt.xlabel("Move Sequence (1 to 100)", fontsize=12)
  plt.ylabel("Accuracy (%) (10-move sliding window)", fontsize=12)
  plt.grid(True, linestyle="--", alpha=0.6)

  # 8. Save the plot as an image file in the same directory
  output_path = script_dir / "ai_learning_curve.png"
  plt.savefig(output_path, dpi=300, bbox_inches="tight")
  print(f"Graph successfully saved to: {output_path}")

  # Display the plot window
  plt.show()