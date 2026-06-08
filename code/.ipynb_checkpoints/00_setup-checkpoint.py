

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q datasets pandas matplotlib nltk

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#regex for text/NLP, numpy, pandas, matplotlib for usual
from datasets import load_dataset #our dataset is from HuggingFace

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 100)

dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train") #our dataset
df = dataset.to_pandas()

print(df.shape)



target_cols = [c for c in df.columns if c.startswith("target_") and pd.api.types.is_bool_dtype(df[c])]


broad_target_cols = [c for c in target_cols if c.count("_") == 1] #this is how you can tell apart the larger "broad" catagories and the specifc
#ones in the dataset
specific_target_cols = [c for c in target_cols if c not in broad_target_cols]


rating_cols = [
    "sentiment", "respect", "insult", "humiliate", "status",
    "dehumanize", "violence", "genocide", "attack_defend",
    "hatespeech", "hate_speech_score"]
rating_cols = [c for c in rating_cols if c in df.columns]

#what the annotators rated

grouped = df.groupby(["comment_id", "text"], as_index=False) #grouping by text to get rid of duplicates

mean_df = grouped[rating_cols].mean()
max_df = grouped[target_cols].max()

#mean is for the mean score, max is because when it comes to classifying who the comment is targeting, this is done with 1 or 0 for columns with each target group, with 1 being it is targeted. Taking the max is the most inclusive way to include all comments that may target a group.

comments = pd.merge(mean_df, max_df, on=["comment_id", "text"])
# I then merged (inner join) these dfs on comment id and text to prevent duplicates. 


comments[target_cols] = comments[target_cols].astype(bool)

#turning to true, false



import nltk #nlp
nltk.download("vader_lexicon")

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

vader_scores = comments["text"].fillna("").apply(lambda x: sia.polarity_scores(x)).apply(pd.Series)
vader_scores = vader_scores.add_prefix("vader_")

#appling the VADER sentiment analysis to the comment text (negative, neutral, positive, and compound)

existing_vader_cols = [col for col in comments.columns if col.startswith('vader_')] #getting rid of existing columns if they have VADER already (AI suggested I do this)
if existing_vader_cols:
    comments = comments.drop(columns=existing_vader_cols)

comments = pd.concat([comments, vader_scores], axis = 1)

print(len(comments))

#concatenating so that there is a column with vader scores


id_cols = ["comment_id", "text", "hate_speech_score", "hatespeech", "vader_neg", "vader_neu", "vader_pos", "vader_compound"]


comments_unique_cols = comments.loc[:, ~comments.columns.duplicated()]

target_long = comments_unique_cols.melt(
    id_vars=id_cols,
    value_vars=specific_target_cols,
    var_name="target",
    value_name="is_target"
)

#this goes through, using melt, and makes a df where each row is a comment x target pair, the unit of analysis I used


target_long = target_long[target_long["is_target"]].copy()


split_df = target_long["target"].str.extract(r"target_(?P<family>[^_]+)_(?P<subgroup>.+)")


target_long["target_family"] = split_df["family"].str.title()
subgroup_clean = split_df["subgroup"].str.replace("_", " ").str.title()


target_long["target_clean"] = target_long["target_family"] + ": " + subgroup_clean

#this was reformatting into clean labels (AI)