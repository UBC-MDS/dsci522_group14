import pandas as pd
import numpy as np

def count_unique_numbers(df, feature_list):
      """_summary_

      Args:
          df (_type_): _description_
          feature_list (_type_): _description_

      Returns:
          _type_: _description_
      """
      
      col_name = []
      counts = []
      for feature in feature_list:
            col_name.append(feature)
            counts.append(len(df[feature].unique().tolist()))
      return pd.DataFrame({"feature":col_name, 
                              "unique_entry_counts": counts}
                              ).sort_values(by="unique_entry_counts", ascending=False)

def generalize_categories(column, df):
      count_df = pd.DataFrame(df[column].value_counts())
      count_df["frequency"] = (count_df["count"] / count_df["count"].sum())
      count_df = count_df.drop(["count"], axis=1)
      count_freq = count_df.to_dict()["frequency"]

      origin_df = df.copy()
      origin_df["frequency"] = origin_df[column].map(count_freq)
      organize_counts = lambda row: "Other" if row["frequency"] <= 0.01 else row[column]
      origin_df[column] = origin_df.apply(organize_counts, axis=1)
      origin_df[column] = origin_df[column].astype("str")
      
      return origin_df