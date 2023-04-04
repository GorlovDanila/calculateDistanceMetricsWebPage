import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def cosine_similarity(set1, set2):
    intersection = set1.intersection(set2)
    vector1 = [list(set1).count(word) for word in intersection]
    vector2 = [list(set2).count(word) for word in intersection]
    # скалярное произведение векторов
    dot_product = sum([vector1[i] * vector2[i] for i in range(len(vector1))])
    # длина векторов
    magnitude1 = sum([vector1[i] ** 2 for i in range(len(vector1))]) ** 0.5
    magnitude2 = sum([vector2[i] ** 2 for i in range(len(vector2))]) ** 0.5
    if dot_product != 0 and magnitude1 != 0 and magnitude2 != 0:
        return dot_product / (magnitude1 * magnitude2)
    else:
        return 0


url = "https://www.sciencenews.org/article/rogue-planets-moons-life"
response = requests.get(url)
text = response.text

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Определение списка стоп-слов для текущего языка
stop_words = set(stopwords.words('english'))

# Токенизация текста
words = word_tokenize(text)

# Приведение слов к нормальной форме
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]

# Удаление стоп-слов из текста
words = [word for word in words if word not in stop_words]

science_keywords = ['science', 'research', 'discovery', 'experiment', 'theory', 'experience', 'cognition', 'proof',
                    'subject', 'knowledge']
sport_keywords = ['sport', 'football', 'basketball', 'tennis', 'swimming', 'figure skating', 'record', 'champion',
                  'competition', 'opponent']
shopping_keywords = ['shopping', 'fashion', 'clothes', 'shoes', 'accessories', 'dresses', 'model', 'show', 'collection',
                     'design']
news_keywords = ['news', 'politics', 'economy', 'weather', 'events', 'sensation', 'scandal', 'meeting', 'elections',
                 'government']

science_set = set(science_keywords)
sport_set = set(sport_keywords)
shopping_set = set(shopping_keywords)
news_set = set(news_keywords)

words_set = set(words)
jaccard_science = jaccard_similarity(words_set, science_set)
jaccard_sport = jaccard_similarity(words_set, sport_set)
jaccard_shopping = jaccard_similarity(words_set, shopping_set)
jaccard_news = jaccard_similarity(words_set, news_set)

cosine_science = cosine_similarity(words_set, science_set)
cosine_sport = cosine_similarity(words_set, sport_set)
cosine_shopping = cosine_similarity(words_set, shopping_set)
cosine_news = cosine_similarity(words_set, news_set)

max_jaccard = max(jaccard_science, jaccard_sport, jaccard_shopping, jaccard_news)
max_cosine = max(cosine_science, cosine_sport, cosine_shopping, cosine_news)

print("Коэффицент Жаккарда: " + str(max_jaccard))
print("Коэффицент Косинуса: " + str(max_cosine))

if max_jaccard == jaccard_science or max_cosine == cosine_science:
    print("Наиболее близкая тематика: наука")
elif max_jaccard == jaccard_sport or max_cosine == cosine_sport:
    print("Наиболее близкая тематика: спорт")
elif max_jaccard == jaccard_shopping or max_cosine == cosine_shopping:
    print("Наиболее близкая тематика: шоппинг")
elif max_jaccard == jaccard_news or max_cosine == cosine_news:
    print("Наиболее близкая тематика: новости")
