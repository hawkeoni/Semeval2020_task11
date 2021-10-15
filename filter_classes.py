from pathlib import Path
from collections import Counter

from tqdm import tqdm

si_labels_path = Path("/home/hawkeoni/Semeval11_propaganda/datasets/train-labels-task1-span-identification")
tc_labels_path = Path("/home/hawkeoni/Semeval11_propaganda/datasets/train-labels-task2-technique-classification")
train_articles_path = Path("/home/hawkeoni/Semeval11_propaganda/datasets/train-articles")
val_articles_path = Path("/home/hawkeoni/Semeval11_propaganda/datasets/dev-articles")
whitelist = {
    "Loaded_Language", 
    "Name_Calling,Labeling", 
    "Repetition", 
    "Exaggeration,Minimisation",
    "Doubt"
}

new_labels = Path("filtered_labels")
new_labels.mkdir(exist_ok=True)

c = Counter()
for si_file in tqdm(si_labels_path.glob("*labels")):
    ti_file = si_file.name.split(".")[0] + ".task2-TC.labels"
    ti_file = Path(ti_file)
    # print(ti_file, ti_file.exists())
    new_si_file = open(new_labels / si_file.name, "w")
    for line in open(tc_labels_path / ti_file):
        num, name, start, end = line.strip().split()
        c[name] += 1
        if name in whitelist:
            new_si_file.write(f"{num}\t{start}\t{end}\n")
    new_si_file.close()

print(c)
"""
article735815503.task1-SI.labels
article754231597.task2-TC.labels
"""
