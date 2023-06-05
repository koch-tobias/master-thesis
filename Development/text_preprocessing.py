import re
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# %%
def prepare_text(designation: str) -> str:
    # transform to lower case
    text = str(designation).upper()

    # Removing punctations
    text = re.sub(r"[^\w\s]", "", text)

    # Removing numbers
    text = ''.join([i for i in text if not i.isdigit()])

    # tokenize text
    text = text.split(" ")

    # Remove predefined words
    predefined_words = ["ZB", "AF", "LI", "RE", "MD", "LL", "TAB", "TB"]
    if len(predefined_words) > 0:
        text = [word for word in text if word not in predefined_words]

    # Remove words with only one letter
    text = [word for word in text if len(word) > 1]

    # remove empty tokens
    text = [t for t in text if len(t) > 0]

    # join all
    prepared_designation = " ".join(text)

    return prepared_designation

# %%
def clean_text(df):
    df["Benennung (bereinigt)"] = df.apply(lambda x: prepare_text(x["Benennung (dt)"]), axis=1)

    return df

# %%
def vectorize_data(data: pd.DataFrame, df_val, timestamp) -> tuple:

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 8))

    X_text = vectorizer.fit_transform(data['Benennung (bereinigt)']).toarray()
    X_test = vectorizer.transform(df_val['Benennung (bereinigt)']).toarray()

    # Store the vocabulary
    vocabulary = vectorizer.get_feature_names_out()

    # Save the vectorizer and vocabulary to files
    os.makedirs(f'../models/lgbm_{timestamp}')
    with open(f'../models/lgbm_{timestamp}/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'../models/lgbm_{timestamp}/vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)

    return X_text, X_test

# %%
def get_vocabulary(column):
    text = ' '.join(column.astype(str))
    words = text.upper().split()
    word_counts = pd.Series(words).value_counts()
    vocabulary = word_counts.index.tolist()

    return vocabulary

