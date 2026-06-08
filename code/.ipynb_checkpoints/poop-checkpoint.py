



#________

import importlib
import re
import numpy as np
import pandas as pd



setup = importlib.import_module("00_setup")


globals().update(vars(setup))


HATE_THRESHOLD = 0.5
MIN_COMMENTS_PER_TARGET = 50


analysis = comments[comments["hate_speech_score"] > HATE_THRESHOLD].copy()

print("Comment-target rows in analysis:", len(analysis))
print("Unique comments in analysis:", analysis["comment_id"].nunique())

hate = target_long[target_long["hate_speech_score"] > HATE_THRESHOLD].copy()


summary = (
    hate.groupby("target_clean")
    .agg(
        n_comments=("comment_id", "nunique"),
        avg_hate_score=("hate_speech_score", "mean"),
        median_hate_score=("hate_speech_score", "median"),
        avg_vader_compound=("vader_compound", "mean"),
        avg_vader_neg=("vader_neg", "mean"),
        avg_vader_pos=("vader_pos", "mean"),
    )
    .reset_index()
)

summary = summary[summary["n_comments"] >= MIN_COMMENTS_PER_TARGET].copy()
summary = summary.sort_values("n_comments", ascending=False)


corr = comments[["hate_speech_score", "vader_compound", "vader_neg", "vader_pos"]].corr()


sample = comments.sample(min(5000, len(comments)), random_state=42)


def contains_term(text_series, term):
    """Case-insensitive whole-word/whole-phrase matching."""
    term = str(term).strip().lower()
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return text_series.fillna("").str.lower().str.contains(pattern, regex=True, na=False)


def word_rate_table(data, terms, min_comments=50):
    """
    Returns mentions per 1,000 comments by target group.
    Each row in `data` should be a comment-target pair.
    """
    base = data.copy()
    target_sizes = base.groupby("target_clean")["comment_id"].nunique().rename("n_comments")

    pieces = []
    for term in terms:
        has_term = contains_term(base["text"], term)
        hits = base.loc[has_term].groupby("target_clean")["comment_id"].nunique()
        rate = (hits / target_sizes * 1000).fillna(0)
        pieces.append(rate.rename(term))

    out = pd.concat(pieces, axis=1).join(target_sizes)
    out = out[out["n_comments"] >= min_comments]
    return out.sort_values("n_comments", ascending=False)


WORDS_TO_SEARCH = [
    "go back",
    "illegal",
    "crime",
    "terrorist",
    "lazy",
    "threat",
    "disease"
]

word_rates = word_rate_table(hate, WORDS_TO_SEARCH, min_comments=MIN_COMMENTS_PER_TARGET)


STOPWORDS = set("""
a an and are as at be been being but by can could did do does doing for from had has have having he her hers
him his i if in into is it its itself just me my of on or our ours she so than that the their theirs them they
this those to too was we were what when where which who whom why will with you your yours about above after
again against all am any because before below between both down during each few further here how more most no nor
not now off once only other out over own same should some such then there these through under until up very
would
""".split())

def tokenize(text):
    text = str(text).lower()
    words = re.findall(r"[a-z']{3,}", text)
    return [w for w in words if w not in STOPWORDS and not w.startswith("http")]

def top_words_for_target(data, target_name, n=25):
    subset = data[data["target_clean"].eq(target_name)]
    tokens = subset["text"].apply(tokenize).explode().dropna()
    return tokens.value_counts().head(n).rename_axis("word").reset_index(name="count")

TARGET_TO_INSPECT = summary.iloc[38]["target_clean"]
print("Inspecting:", TARGET_TO_INSPECT)

def distinctive_words(data, target_name, min_count=10, n=25):
    target_texts = data.loc[data["target_clean"].eq(target_name), "text"]
    other_texts = data.loc[~data["target_clean"].eq(target_name), "text"]

    target_tokens = target_texts.apply(tokenize).explode().dropna()
    other_tokens = other_texts.apply(tokenize).explode().dropna()

    target_counts = target_tokens.value_counts()
    other_counts = other_tokens.value_counts()

    vocab = target_counts.index.union(other_counts.index)

    table = pd.DataFrame({
        "target_count": target_counts.reindex(vocab, fill_value=0),
        "other_count": other_counts.reindex(vocab, fill_value=0),
    })

    target_total = table["target_count"].sum()
    other_total = table["other_count"].sum()
    vocab_size = len(table)

    table["target_rate"] = (table["target_count"] + 1) / (target_total + vocab_size)
    table["other_rate"] = (table["other_count"] + 1) / (other_total + vocab_size)
    table["log2_rate_ratio"] = np.log2(table["target_rate"] / table["other_rate"])

    table = table[table["target_count"] >= min_count]
    return (
        table.sort_values("log2_rate_ratio", ascending=False)
             .head(n)
             .reset_index()
             .rename(columns={"index": "word"})
    )


family_summary = (
    hate.groupby("target_family")
    .agg(
        n_comments=("comment_id", "nunique"),
        avg_hate_score=("hate_speech_score", "mean"),
        avg_vader_compound=("vader_compound", "mean"),
        avg_vader_neg=("vader_neg", "mean"),
    )
    .sort_values("n_comments", ascending=False)
)


multi_target_comment_ids = target_long["comment_id"].value_counts()
multi_target_comment_ids = multi_target_comment_ids[multi_target_comment_ids > 1].index

multi_target_df = target_long[target_long["comment_id"].isin(multi_target_comment_ids)].copy()

print(f"Number of comments targeting multiple groups: {len(multi_target_comment_ids)}")
print(f"Total entries in multi_target_df: {len(multi_target_df)}")

targets_per_comment = multi_target_df.groupby("comment_id")["target_clean"].apply(list)

all_pairs = []
for targets_list in targets_per_comment:
    sorted_targets = sorted(targets_list)
    for i in range(len(sorted_targets)):
        for j in range(i + 1, len(sorted_targets)):
            all_pairs.append((sorted_targets[i], sorted_targets[j]))

if all_pairs:
    co_occurrence_temp_df = pd.DataFrame(all_pairs, columns=["target_1", "target_2"])
    co_occurrence_df = co_occurrence_temp_df.groupby(["target_1", "target_2"]).size().reset_index(name="count")
else:
    co_occurrence_df = pd.DataFrame(columns=["target_1", "target_2", "count"])

co_occurrence_df = co_occurrence_df.sort_values("count", ascending=False).reset_index(drop=True)

def get_co_targeted_groups(co_occurrence_df, main_target, n=10):
    filtered_df = co_occurrence_df[
        (co_occurrence_df["target_1"] == main_target) |
        (co_occurrence_df["target_2"] == main_target)
    ].copy()

    filtered_df["other_target"] = filtered_df.apply(
        lambda row: row["target_2"] if row["target_1"] == main_target else row["target_1"],
        axis=1
    )

    result = filtered_df.groupby("other_target")["count"].sum().sort_values(ascending=False)
    return result.head(n)


target_counts_per_comment = target_long.groupby("comment_id")["target_clean"].nunique()

single_target_comment_ids = target_counts_per_comment[target_counts_per_comment == 1].index
multi_target_comment_ids_v2 = target_counts_per_comment[target_counts_per_comment > 1].index

single_targeted_df = target_long[target_long["comment_id"].isin(single_target_comment_ids)]
single_targeted_counts = single_targeted_df["target_clean"].value_counts()

multi_targeted_df = target_long[target_long["comment_id"].isin(multi_target_comment_ids_v2)]
multi_targeted_counts = multi_targeted_df["target_clean"].value_counts()

targeting_summary = pd.DataFrame({
    "targeted_alone": single_targeted_counts,
    "co_targeted": multi_targeted_counts
}).fillna(0).astype(int)

targeting_summary["total_mentions"] = targeting_summary["targeted_alone"] + targeting_summary["co_targeted"]
targeting_summary["proportion_co_targeted"] = targeting_summary["co_targeted"] / targeting_summary["total_mentions"]
targeting_summary = targeting_summary.sort_values("total_mentions", ascending=False)
