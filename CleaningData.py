#Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('speed_data_data.csv')
df.head(5)

# Basic Information of Dataset
print("\n<<Basic Information of Dataset>>")
print(df.shape)
print(df.index)
print(df.columns)

###### Clean Data ######

# Can not found goal's meaing -> drop
df = df.drop(columns = ['goal'])

# Missing Values
print('\n<<Missing Data>>')
print(df.isnull().sum())

# Career
# 주요 카테고리로 그룹화하기 위한 사전 정의
# 성능에 따라 Category를 더 추가하거나 / Other row를 drop해야할 수 있다.
# 사전 정의된 카테고리 매핑
career_mapping = {
    'Legal': ['lawyer', 'lawyer/policy work', 'law', 'corporate lawyer', 'lawyer or professional surfer', 'ip law', 'tax lawyer', 'lawyer/gov.position', 'attorney', 'corporate attorney', 'legal', 'lawyer', 'lawyer'],
    'Finance': ['economist', 'financial services', 'investment banking', 'finance', 'asset management', 'trading', 'hedge fund', 'private equity', 'investment management', 'investment banker', 'investment', 'banker', 'capital markets', 'fixed income sales & trading', 'financial service', 'wall street economist'],
    'Healthcare': ['doctor', 'physician', 'pediatrics', 'clinical psychologist', 'clinical psychology', 'medicine', 'medical sciences', 'psychiatrist', 'dentist', 'healthcare', 'dietician', 'nutritionist', 'biostatistics', 'speech pathologist', 'health/nutrition oriented social worker', 'physician scientist'],
    'Education': ['professor', 'teacher', 'academic', 'academia', 'school psychologist', 'elementary education teaching', 'school counseling', 'university professor', 'education administration', 'academic work', 'academic or consulting', 'education policy analyst', 'college professor', 'academic physician', 'school psychologist', 'professor of education'],
    'Engineering': ['engineer', 'mechanical engineering', 'industrial scientist', 'science', 'engineering', 'software engineer', 'network engineer', 'cto', 'ceo', 'research scientist', 'biotech', 'pharmaceuticals', 'ph.d. electrical engineering', 'informatics', 'engineer or ibanker or consultant'],
    'Technology': ['tech professional', 'informatics', 'software engineer', 'cto', 'ceo', 'research scientist', 'biotech', 'pharmaceuticals', 'computer science', 'it', 'technology', 'data scientist', 'program development / policy work', 'industrial scientist'],
    'Arts': ['artist', 'musician', 'writer', 'film', 'filmmaker', 'screenwriter', 'producer', 'entertainment', 'journalist', 'poet', 'arts', 'art educator', 'art management', 'music production', 'writing', 'acting', 'creative', 'media'],
    'Business': ['ceo', 'entrepreneur', 'business', 'management', 'consulting', 'marketing', 'business management', 'manager', 'sales', 'business consulting', 'business management', 'strategy', 'operations', 'brand management', 'corporate finance', 'business/law', 'business - investment management', 'business consulting', 'marketing and media'],
    'Consulting': ['consultant', 'management consultant', 'strategy consultant', 'advisory', 'consulting', 'business consulting', 'management consulting'],
    'Science': ['scientist', 'researcher', 'academic', 'biologist', 'chemist', 'physicist', 'epidemiologist', 'neuroscientist', 'science', 'research', 'research/academia', 'research scientist', 'research position', 'scientific research', 'researcher in sociology'],
    'Social Work': ['social worker', 'social work', 'counselor', 'therapist', 'psychologist', 'speech pathologist', 'health policy', 'social services', 'child rights', 'clinical social worker', 'social worker.... clinician', 'social work policy', 'community work', 'human rights'],
    'Government': ['government', 'public service', 'politician', 'diplomat', 'public policy', 'policy work', 'civil servant', 'international affairs', 'foreign service', 'public finance', 'political development', 'policy advisor', 'public administration', 'homeland defense'],
    'Entertainment': ['entertainment', 'actor', 'actress', 'producer', 'filmmaker', 'director', 'screenwriter', 'tv', 'film', 'entertainment industry', 'comedienne', 'playing music', 'music industry'],
    'Sports': ['athlete', 'sports', 'coach', 'trainer', 'professional athlete', 'pro beach volleyball', 'sports industry', 'boxing champ'],
    'Marketing': ['marketing', 'advertising', 'brand management', 'media marketing', 'strategy and business development', 'media management'],
    'Real Estate': ['real estate', 'property management', 'real estate consulting', 'real estate/private equity'],
    'Unknown': ['?', 'unknown', "don't know", 'dont know yet', 'not sure', 'undecided', 'if only i knew', 'still wondering', 'no idea', 'who knows', 'am not sure', 'tba', 'unsure', 'unknown', 'not sure yet']
}

# Convert all data to lower case
df['career'] = df['career'].str.lower()

# Mapping Function
def map_career(career):
    for category, keywords in career_mapping.items():
        if any(keyword in career for keyword in keywords):
            return category
    return 'Other'  # Specify as 'Other' if not mapped

# Apply mapping
df['career'] = df['career'].apply(lambda x: 'Unknown' if pd.isna(x) else map_career(str(x)))


# Drop row Unknown value 
df = df[df['career'] != 'Unknown']

# Result of cleaning career column
print("\n<<Result of cleaning 'career' column>")
unique_values = df['career'].unique()
print(unique_values)
print(df['career'].value_counts())

# missing value substitution function for 'income' column by group average value
def fill_na_with_mean(df, group_col, target_col):
    df[target_col] = df.groupby(group_col)[target_col].transform(lambda x: x.fillna(x.mean()))
    return df

# "Income" Missing value replacement
df = fill_na_with_mean(df, 'career', 'income')

# Processing to replace the  missing values
df['age'].fillna(df['age'].mean(), inplace = True)
df['attr'].fillna(df['attr'].mean(), inplace=True)
df['sinc'].fillna(df['sinc'].mean(), inplace=True)
df['intel'].fillna(df['intel'].mean(), inplace=True)
df['fun'].fillna(df['fun'].mean(), inplace=True)
df['amb'].fillna(df['amb'].mean(), inplace=True)
df['shar'].fillna(df['shar'].mean(), inplace=True)
df['like'].fillna(df['like'].mean(), inplace=True)
df['prob'].fillna(df['prob'].mean(), inplace=True)
df = df.dropna(subset=['met'])
# Converting 'met' column values to binary variables
# 0 converts to "meet" and other values to "never met"
df['met'] = df['met'].apply(lambda x: 0 if x == 0 else 1)

# Check is there any Missing Data
print('\n<Final Missing Data>')
print(df.isnull().sum())

# Result of Cleaining data
print("\n")
print(df.head())

# One-hot encoding by get_dummies function in pandas library
df= pd.get_dummies(df, columns=['career'])
print('\n<Data after Encoding>')
print(df.head(20))

# Detect outliers with z-score
def find_outlier(data, threshold=5):
    z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
    outliers = np.where(z_scores > threshold)
    print("Number of ouliers when threshold = {}: ".format(threshold), len(np.unique(outliers[0])))
    return np.unique(outliers[0])
    
test = df[['age', 'income', 'dec', 'attr', 'sinc', 'intel', 'fun', 'amb',   
       'shar', 'like', 'prob']]

outliers = find_outlier(test, threshold=3)

# Delete the outliers
df = df.drop(df.index[outliers])

print("Length after deleting outliers: ", len(df.index))

# Show the Dataframe without oulier
df.hist(figsize=(10, 9))
plt.tight_layout()
plt.show()

# Split the target and features
X = df.drop(columns=['dec'])
y = df['dec']

# Scaling with MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

print("\n<Data after Scaling>")
print(X.head(10))

# Saved to other file(If it needed)
df.to_csv('cleaned_speed_data.csv', index=False)
