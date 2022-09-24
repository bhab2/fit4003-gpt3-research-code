import requests
from bs4 import BeautifulSoup
import csv
import sys

BASE_URL = 'https://capterra.com.au'
SEARCH_URL = BASE_URL + '/search/?q='
PAGE_URL = '?page='

print("Enter the app name: ")
appName = input('> ').lower()
fileName = appName.lower().replace(' ', '-')

html_text = requests.get(SEARCH_URL + appName.replace(' ', '-')).text

print('')
print(f'Looking for reviews of {appName} on Capterra...')
print('')

soup_main = BeautifulSoup(html_text, 'html.parser')
search_results = soup_main.find_all('img')

correct_result = None
searched_app_link = None
reviewCount = 0

header = ['Review Date', 'Review Title', 'Review Summary', 'Review Comments']

# find link for the searched app
for result in search_results:
    if appName.casefold() == result.get('alt').casefold():
        correct_result = result

try:
    for parent in correct_result.parents:
        if parent.name == 'a':
            searched_app_link = BASE_URL + parent.get('href')
            searched_app_link = searched_app_link.replace('software', 'reviews')  # change link to reviews link
except AttributeError:
    print('The app could not be found on Capterra!')
    sys.exit(1)

# get reviews and write into a csv file
with open(f'{fileName}.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)

    # go through all pages of the website to get reviews
    for page in range(1, 100):

        app_html_text = requests.get(searched_app_link + PAGE_URL + str(page)).text
        soup = BeautifulSoup(app_html_text, 'lxml')
        reviews = soup.find_all('div', class_='col-lg-7')

        for index, review in enumerate(reviews):
            reviewCount += 1

            review_date = review.find('span', class_='ms-2').text.strip()  # gets the date of the review
            # get the review title
            review_title = review.find('h3', class_='h5 fw-bold').text.strip()
            review_summary = None
            review_comments = []

            # go through all reviews
            for p in review.find_all('p'):

                if p.find('span'):
                    p.span.decompose()  # removes span tags within p tags
                    review_summary = p.text.strip()
                else:
                    review_comments.append(p.text.strip())

            for comment in review_comments:
                if comment.lower() == 'pros:' or comment.lower() == 'cons:':
                    review_comments.remove(comment)

            writer.writerow([review_date, review_title, review_summary, review_comments])

print(f'{reviewCount} reviews to added csv file named {fileName}.csv')
