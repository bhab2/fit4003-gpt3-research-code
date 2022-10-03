import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from transformers import pipeline

pipe = pipeline('sentiment-analysis')

print('Enter the file(.csv) name for analysis: ')
fileName = input('> ').lower()

print('')
print('')

print(f'Reading {fileName}...')
df = pd.read_csv(fileName)
df.head()
print('')
print('')

dataArray = []
data = ' '.join(text for text in df.Summary if text != ' ')

for text in df.Summary:
    if text != ' ':
        dataArray.append(text)

print('This may take a moment. Analysing review sentiments...')
print('')
sentiments = pipe(dataArray)
print('')
print('')

print('Calculating Positive and Negative reviews...')
print('')
total = len(dataArray)
positive = 0
negative = 0

for sentiment in sentiments:
    if sentiment['label'] == 'POSITIVE':
        positive += 1
    else:
        negative += 1

positivePercent = (positive / total) * 100
negativePercent = (negative / total) * 100

plt.bar(['positive', 'negative'], [positivePercent, negativePercent], color=['green', 'red'])
plt.title('Sentiment Analysis')
plt.xlabel('Review Sentiment')
plt.ylabel('Percent found')
plt.show()

print('')
print('Bar Chart generated for user sentiments!')
print('Close Bar Chart to proceed with Word Cloud...')
print('')
print('')

print('Generating Word Cloud for Review Summary...')
print('Close chart to end script...')
print('')
word_cloud = WordCloud(
    width=3000,
    height=2000,
    random_state=1,
    background_color='salmon',
    colormap='Pastel1',
    collocations=False,
    stopwords=STOPWORDS
).generate(data)

plt.imshow(word_cloud)
plt.axis("off")
plt.show()

print('Word Cloud generated for review summary!')
