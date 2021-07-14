import pandas as pd
import numpy as np

# CSV file should be in the same directory as the current one
BOLLYWOOD_PATH = "bollywood.csv"
bollywood = pd.read_csv(BOLLYWOOD_PATH)
bollywood.drop_duplicates(inplace=True)

C = 5.5
m = 2500

#Adding modified rating
bollywood["rating"] = ((bollywood["imdb_votes"])/(bollywood["imdb_votes"] + m))*bollywood["imdb_rating"] + (m/(bollywood["imdb_votes"] + m))*C

#Sorting using the new rating
top_10 = bollywood.sort_values('rating', ascending = False).head(10)
top_10.index = np.arange(1, len(top_10)+1)
print(top_10)