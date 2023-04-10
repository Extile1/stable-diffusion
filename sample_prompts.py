import pandas as pd
import subprocess

#get prompts from diffusion_db
#test how good the laion model is more
df = pd.read_parquet('laion_high_res_5B.parquet')

df = df[df["LANGUAGE"] == "en"]
df = df.sample(100)["TEXT"]
prompts = list(df)

for prompt in prompts:
    command = ["python", "scripts/txt2img.py", "--prompt" , f'"{prompt}"', "--plms", "--n_samples", "1"]
    subprocess.run(command)

with open('./prompts.txt', 'w') as f:
    for prompt in prompts:
        f.write(prompt + "\n")