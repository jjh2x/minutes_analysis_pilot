from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import pandas as pd
okt = Okt()
vectorizer = TfidfVectorizer()

# 바탕화면의 '메모장 크롤링 테스트.txt' 파일 불러오기.
f = open("C:/Users/jjh37/Desktop/메모장 크롤링 테스트.txt", 'r', encoding='UTF8')
# 메모장 파일의 내용을 줄 단위(엔터 기준)로 나누어 'sentences' 변수에 리스트 형태로 저장.
sentences = f.readlines()
f.close()


# =========================================================================
# Vectorizer의 argument인 tokenizer에 KoNLPy의 pos 함수로 대체.

class MyTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger

    def __call__(self, sent):
        pos = self.tagger.pos(sent)
        clean_words = []            # 정제된 단어 리스트
        for word in pos:
            # word[1]은 품사를 의미하며, 여기서는 조사, 문장기호, 접두사, Foreign('\n'을 빼주기 위함)인 것은 제외시킴.
            if word[1] not in ['Josa', 'Punctuation', 'Suffix', 'Foreign']:
                if len(word[0]) >= 2 :      # 한 글자인 단어들도 의미가 없는 경우가 많으므로 일단 제외.
                    clean_words.append(word)
        return clean_words
# =========================================================================

my_tokenizer = MyTokenizer(Okt())
tfidf_Vectorizer = TfidfVectorizer(tokenizer=my_tokenizer, min_df=3)    # df 값(단어가 몇 문장들에서 등장하였는지)을 최소 3으로 설정.
X = tfidf_Vectorizer.fit_transform(sentences).toarray()
print(X.shape)      # X(2차원 배열)의 행,열 수를 출력.
print(tfidf_Vectorizer.vocabulary_, "\n")     # 각 단어들이 X에서 몇번째 열(인덱스 값)에 해당하는지 출력.

# =========================================================
# pandas를 활용하여 각 단어들의 각 문장에서의 tf-idf 값들을 모두 더하고, 내림차순으로 정렬하여 상위 n개 출력.
count = X.sum(axis=0)           # 2차원 배열 X에서 각 열을 기준으로 합을 구함. (각 단어들의 '최종' tf-idf 값으로 간주.)
word_count = pd.DataFrame({
    '단어' : tfidf_Vectorizer.get_feature_names(),
    '빈도' : count.flat})
sorted_df = word_count.sort_values('빈도', ascending=False)
print(sorted_df.head(10))
# =========================================================




