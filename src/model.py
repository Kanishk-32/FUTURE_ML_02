from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def train_model(text_data, target):
    
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(text_data)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.2, random_state=42
    )

    
    model = MultinomialNB()
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred