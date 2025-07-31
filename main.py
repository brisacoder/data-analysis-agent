import asyncio
import textwrap
from langchain_sandbox import PyodideSandbox

code = r"""
import httpx
import pandas as pd
# matplotlib.use("Agg")

url = "https://raw.githubusercontent.com/Walendziak1912/Simpson-DataSets/master/simpsons_episodes.csv"

# Use httpx (requests won't work in Pyodide)
response = httpx.get(url)
response.raise_for_status()

# Save to file
with open("simpsons.csv", "w", encoding="utf-8") as f:
    f.write(response.text)


df = pd.read_csv("simpsons.csv")

# Drop rows where 'us_viewers_in_millions' is NaN
df_clean = df.dropna(subset=['us_viewers_in_millions'])
# Sort by 'us_viewers_in_millions' descending, get top N
top_eps = df_clean.sort_values(by='us_viewers_in_millions', ascending=False).head(10)
# Iterate and pretty print each episode's info as dict
result = ""
for i, row in top_eps.iterrows():
    result += "\\n" + f"Episode #{i+1} (Row index {i})"
    result += f"{row.to_dict()}"

print(result)
"""

async def main():

    # Create a sandbox instance
    sandbox = PyodideSandbox(
        # Allow Pyodide to install python packages that
        # might be required.
        allow_net=True,
    )

    # Execute Python code
    output = await sandbox.execute(code)
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
