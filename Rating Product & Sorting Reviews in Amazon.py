
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# Business Problem
###################################################

# One of the most significant problems in e-commerce is the accurate calculation of post-purchase ratings given to products.
# The solution to this problem means providing more customer satisfaction for e-commerce websites, highlighting products for sellers, and ensuring a seamless shopping experience for buyers. 
# Another challenge is the accurate sorting of product reviews. Since misleading reviews can directly impact the sales of a product, it may result in both financial loss and customer attrition. 
# Addressing these two fundamental problems will not only increase sales for e-commerce sites and sellers but also allow customers to complete their purchasing journey smoothly."

###################################################
# Dataset Overwiew
###################################################

# This dataset containing Amazon product data includes various metadata along with product categories. 
# It includes user ratings and reviews for the product that received the highest number of reviews in the Electronics category.

# Değişkenler:
# reviewerID: user's ID
# asin: product's ID
# reviewerName: User name
# helpful: Useful rating level
# reviewText: Assessment
# overall: Product rating
# summary: Assessment summary
# unixReviewTime: Assessment time
# reviewTime: Assessment time-raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: The number of days since the review
# total_vote: The number of votes given to the review

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_counts",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",500)
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.float_format",lambda x: "%.5f" % x)


###################################################
# Tsk 1:Calculate the Average Rating based on the recent reviews and compare it with the existing Average Rating.
###################################################


# The shared dataset includes user ratings and reviews for a product. 
# The goal of this task is to evaluate the given ratings by weighting them based on the date.
# A comparison between the initial average rating and the date-weighted rating obtained needs to be performed."


###################################################
# Step 1: Read dataset and calculate the average point of product.  
###################################################

df=pd.read_csv("Datasets/amazon_review.csv")
df.head(20)

df["overall"].mean()

###################################################
# Adım 2: Calculate the weighted point average per time.
###################################################

df.dtypes

df["reviewTime"]=pd.to_datetime(df["reviewTime"])
df.dtypes

current_date = pd.to_datetime('2021-02-10 0:0:0')

df["days"]=(current_date-df["reviewTime"]).dt.days

df.describe().T

df["days"].sort_values(ascending=True)
df.head(20)

q1= df["days"].quantile(0.25)
q2= df["days"].quantile(0.50)
q3= df["days"].quantile(0.75)
df["days"].describe().T

def time_based_weighted_average(dataframe, w1=18, w2=22, w3=28, w4=32):
    return dataframe.loc[(dataframe["days"] <= q1), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > q1) & (dataframe["days"] <= q2), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > q2) & (dataframe["days"] <= q3), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > q3),"overall"].mean() * w4 / 100


time_based_weighted_average(df)

# 4.568060108268714

###################################################
# Task 2: For product, indicate 20 review to be demonstrated in product detail page.
###################################################

###################################################
# Step 1. Generate teh varibale of "helpful_no"
###################################################

# Note:
# total_vote is a total up-down number given to a comment.
# up means "helpful".
# There isn't the variable of "helpful_no" in dataset. It should be generated through already existing variables.  

df["helpful_no"] =df["total_vote"] - df["helpful_yes"]
df.head()

###################################################
# Step 2. Calculate score of score_pos_neg_diff, score_average_rating and wilson_lower_bound. Add them to dataset.
###################################################

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x:wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


##################################################
# Step 3. Indicate 20 comments and evaluate those results.
###################################################

df[["overall","helpful_yes","total_vote","days","helpful_no", "score_pos_neg_diff", "score_average_rating","wilson_lower_bound"]]\
.sort_values("wilson_lower_bound", ascending= False).head(20)
