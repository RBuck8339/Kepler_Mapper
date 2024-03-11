import pandas as pd

df = pd.read_csv('OGB_Dataset/ogbg_molhiv/raw/edge-feat.csv.gz', compression='gzip')
print("Edge Feat Head")
print(df.columns)
print(df.head)
 
df = pd.read_csv('OGB_Dataset/ogbg_molhiv/raw/edge.csv.gz', compression='gzip')
print("Edge head")
print(df.head)

df = pd.read_csv('OGB_Dataset/ogbg_molhiv/raw/graph-label.csv.gz', compression='gzip')
print("Graph Label Head")
print(df.head)

df = pd.read_csv('OGB_Dataset/ogbg_molhiv/raw/node-feat.csv.gz', compression='gzip')
print("Node Feat Head")
print(df.head)

# The number of edges for 
df = pd.read_csv('OGB_Dataset/ogbg_molhiv/raw/num-edge-list.csv.gz', compression='gzip')
print("Num Edge List Head")
print(df.head)

# Literally is just the number of nodes per each molecule
df = pd.read_csv('OGB_Dataset/ogbg_molhiv/raw/num-node-list.csv.gz', compression='gzip')
print("Num Node List Head")
print(df.head)
