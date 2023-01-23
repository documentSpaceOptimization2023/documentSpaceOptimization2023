
import pandas as pd

# path1 = "C:\\Users\\user\\Desktop\\results_simple_greedy.csv"
# path2 = "C:\\Users\\user\\Desktop\\results_heuristic_greedy.csv"
# path3 = "C:\\Users\\user\\Desktop\\results_model_greedy.csv"
# df1 = pd.read_csv(path1) 
# df2 = pd.read_csv(path2) 
# df3 = pd.read_csv(path3) 

# df = df1.merge(df2, on='Name')
# df = df.merge(df3, on='Name')


# df.to_csv('C:\\Users\\user\\Desktop\\new_finals.csv', index=False) #change here
#print(df)

df = pd.read_csv('C:\\Users\\user\\Desktop\\new_finals.csv') 


df = df[df.Reduced == True]

print(df)

df.to_csv('C:\\Users\\user\\Desktop\\new_finals2.csv', index=False) #change here
