# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 22:25:03 2023

@author: suhai
"""
import numpy as np
import pandas as pd
import ast
from collections import Counter

foundations = {"binding": ["individual", "binding"], "moral": ["moral"],
               "full": ["care", "fairness", "loyalty", "authority", "purity"],
               "complete": ["care", "harm", "fairness", "cheating", "loyalty", "betrayal", "authority", "subversion",
                            "purity", "degradation"]}
foundations_dict = {"binding": {"harm": "individual", "care": "individual", "degradation": "binding", 
                    "purity": "binding", "betrayal": "binding", "loyalty": "binding", 
                    "subversion": "binding", "authority": "binding",
                    "cheating": "individual", "fairness": "individual", 
                    "non-moral": "non-moral", "nm": "non-moral"},
                    
                    "moral": {"harm": "moral", "care": "moral", "degradation": "moral", 
                                    "purity": "moral", "betrayal": "moral", "loyalty": "moral", 
                                    "subversion": "moral", "authority": "moral",
                                    "cheating": "moral", "fairness": "moral", 
                                    "non-moral": "non-moral", "nm": "non-moral"},
                    "full": {"harm": "care", "care": "care", "degradation": "purity", 
                                        "purity": "purity", "betrayal": "loyalty", "loyalty": "loyalty", 
                                        "subversion": "authority", "authority": "authority",
                                        "cheating": "fairness", "fairness": "fairness", 
                                        "non-moral": "non-moral", "nm": "non-moral"},
                    "complete": {"harm": "harm", "care": "care", "degradation": "degradation", 
                                        "purity": "purity", "betrayal": "betrayal", "loyalty": "loyalty", 
                                        "subversion": "subversion", "authority": "authority",
                                        "cheating": "cheating", "fairness": "fairness", 
                                        "non-moral": "non-moral", "nm": "non-moral"}
                    }


def transformData(mode = "moral"):
    df = pd.read_csv("mftc_data/mftc_cleaned.csv", index_col = 0).drop_duplicates("tweet_id") #drop duplicates!
    
    # find majority vote of annotators:  
    # create raw dataframe
    df2 = df.iloc[:, :3].copy().reset_index(drop = True)
    # transform anntotations to list of dicts
    df2.annotations = df2.annotations.apply(lambda x: ast.literal_eval(x))
    # transform to list of moral foundations (combine vices/virtues), count each category only once per annotator (dont count duplicates!)
    df2["cleaned_annotations"] = df2.annotations.apply(lambda x: 
                               list([a for l in x for a in set(map(foundations_dict[mode].get, l["annotation"].split(",")))]))
    # get number of  annotators
    df2["nr_annotators"] = df2.annotations.apply(lambda x: len([l["annotator"] for l in x]))
    
    #count a tweet for foundation if at least half the annotators choose it and if "non-moral" 
    for foundation in foundations[mode]:
        df2[foundation] = df2.apply(lambda x: 1 if (x["cleaned_annotations"].count(foundation)/x["nr_annotators"] >= 0.5) and 
                                    not all([Counter(x["cleaned_annotations"])["non-moral"] > value 
                                        for key, value in Counter(x["cleaned_annotations"]).items() 
                                        if key != "non-moral"]) else 0, axis = 1)
    
    #Save file
    df2.to_csv("mftc_data/mftc_cleaned_combined_" + mode + ".csv")
    return 0


transformData("full")













