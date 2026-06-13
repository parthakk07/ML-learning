# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv(
    "/home/parthak/Videos/autoYT/try_eda/genz_college_admission_prediction.csv"
)
df.sample(5)

# %%
df.info()

# %%
df.isnull().sum()
# %%
df.describe()
# %%
df.duplicated().sum()

# %%
df.shape

# %%
df.corr()
# so making data less so i can do eda my laptop crashed twice lol
# %%
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.1, random_state=42)

# %%


# %%
df = pd.read_csv("/home/parthak/Videos/autoYT/try_eda/test.csv")

# %%
df.shape
# %%
df.info()
# %%
df.describe()
# %%
df.isnull().sum()
# %%
df.duplicated().sum()
# %%
# univariate analysis
#  categorical column
sns.countplot(x=df["admission_status"])

# %%
df["admission_status"].value_counts().plot(kind="pie", autopct="%.2f")

# %%
df["gender"].value_counts().plot(kind="pie", autopct="%.2f")

# %%
sns.countplot(x=df["leadership_positions"])

# %%
df["leadership_positions"].value_counts().plot(kind="pie", autopct="%.2f")
# %%
# numerical columns
sns.histplot(x=df["age"], kde=True)
df["age"].skew()
# %%
sns.displot(x=df["age"])
# %%
sns.boxplot(x=df["age"])
# %%
sns.histplot(x=df["state"])
# %%
sns.countplot(x=df["state"])
# %%
sns.displot(x=df["family_income"], kde=True)
df["family_income"].skew()
# %%
sns.boxplot(x=df["family_income"])
# %%
sns.displot(x=df["high_school_gpa"], kde=True)
df["high_school_gpa"].skew()
# %%
sns.displot(x=df["sat_score"])
df["sat_score"].skew()
# %%
sns.displot(x=df["act_score"], kde=True)
df["act_score"].mean()
df["act_score"].skew()
# %%
sns.displot(x=df["attendance_rate"], kde=True)
# %%
sns.displot(x=df["ap_courses"])
# %%
sns.boxplot(x=df["ap_courses"])
# %%
sns.displot(x=df["extracurricular_count"], kde=True)
df["extracurricular_count"].skew()
# %%
sns.boxplot(x=df["extracurricular_count"])
# %%
sns.displot(x=df["volunteer_hours"], kde=True)
# %%
sns.boxplot(x=df["volunteer_hours"])
# %%
sns.displot(x=df["coding_projects"])
# %%
sns.boxplot(x=df["coding_projects"])
# %%
sns.displot(x=df["social_media_hours"], kde=True)
# %%
sns.boxplot(x=df["social_media_hours"])
# %%
sns.displot(x=df["online_certifications"], kde=True)
# %%
sns.countplot(x=df["online_certifications"])
# %%
sns.displot(x=df["essay_score"], kde=True)
# %%
sns.histplot(x=df["essay_score"], kde=True)
# %%
sns.boxplot(x=df["essay_score"])
# %%
sns.displot(x=df["recommendation_score"])
# %%
sns.boxplot(x=df["recommendation_score"])
# %%
sns.boxplot(x=df["interview_score"])
# %%
sns.histplot(x=df["interview_score"], kde=True)
# %%
# multivarite analysis
sns.scatterplot(x=df["family_income"], y=df["age"], size=df["high_school_gpa"])
# %%
sns.scatterplot(x=df["high_school_gpa"], y=df["age"])
# %%
sns.scatterplot(x=df["high_school_gpa"], y=df["sat_score"])
# %%
sns.scatterplot(x=df["high_school_gpa"], y=df["act_score"])
# %%
sns.scatterplot(x=df["high_school_gpa"], y=df["volunteer_hours"])
# %%
sns.scatterplot(x=df["sat_score"], y=df["act_score"])
# %%
sns.scatterplot(x=df["sat_score"], y=df["volunteer_hours"], hue=df["admission_status"])
# %%
sns.scatterplot(
    x=df["sat_score"], y=df["social_media_hours"], style=df["admission_status"]
)
# %%
sns.scatterplot(x=df["sat_score"], y=df["interview_score"])
# %%
sns.scatterplot(
    x=df["sat_score"],
    y=df["act_score"],
    size=df["coding_projects"],
    hue=df["admission_status"],
)
# %%
sns.displot(x=df["age"], y=df["admission_status"])
# %%
sns.pairplot(df)
# %%
from pandas_profiling import ProfileReport

prof = ProfileReport(df)
prof.to_file(output_file="output.html")
# %%
