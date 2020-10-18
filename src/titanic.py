import pandas as pd
import re
import nltk

def name_preprocessing(df_names, name_col = 'Name'):
    # preprocess name to return cleaned vector, title and last name
    df_names.loc[:, 'Last Name'] = df_names[name_col].str.split(',').str[0].str.lower().str.strip()
    df_names.loc[:, 'Last Name'] = [re.sub('[^a-z]+', ' ', str(s)) for s in df_names['Last Name']]
    df_names.loc[:, 'Title'] = df_names['Name'].str.split(',').str[1].str.split('.').str[0].str.lower().str.strip()
    df_names.loc[:, 'Name Vec'] = [re.sub('[^a-z]+', ' ', str(s)) for s in df_names[name_col].str.lower()]
    df_names.loc[:, 'Name Vec'] = [re.sub(' +', ' ', str(s)) for s in df_names['Name Vec']]
    df_names.loc[:, 'Name Vec'] = df_names['Name Vec'].apply(lambda x: nltk.word_tokenize(x))
    return df_names

def sex_numeric(df_sex, sex_col = 'Sex'):
    df_sex.loc[df_sex[sex_col].str.lower().str.strip() == 'male', 'Sex'] = 0
    df_sex.loc[df_sex[sex_col].str.lower().str.strip() == 'female', 'Sex'] = 1
    df_sex[sex_col] = pd.to_numeric(df_sex[sex_col])
    return df_sex
