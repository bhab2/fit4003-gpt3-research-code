import os
import pandas as pd
import tweepy
from tweepy import OAuthHandler
from datetime import datetime

CONSUMER_KEY = 'consumer key here'
CONSUMER_SECRET = 'consumer secret key here'

ACCESS_KEY = 'access key here'
ACCESS_START = 'access start key here'

# Twitter API Authentication
auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_KEY, ACCESS_START)
api = tweepy.API(auth)


def scraptweets(keywords, numTweets, numRuns):

    db_tweets = pd.DataFrame(columns=['username', 'acctdesc',
                                      'retweetcount', 'text', 'hashtags'])

    for _ in range(0, numRuns):

        tweets = tweepy.Cursor(api.search_tweets, q=keywords, lang="en",
                               tweet_mode='extended').items(numTweets)

        tweet_list = [tweet for tweet in tweets]
        noTweets = 0

        for tweet in tweet_list:
            username = tweet.user.screen_name
            acctdesc = tweet.user.description
            retweetcount = tweet.retweet_count
            hashtags = tweet.entities['hashtags']

            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:
                text = tweet.full_text

            ith_tweet = [username, acctdesc, retweetcount, text, hashtags]

            db_tweets.loc[len(db_tweets)] = ith_tweet

            noTweets += 1

        print('no. of tweets scraped is {}'.format(noTweets))

    to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')

    path = os.getcwd()
    filename = path + "/" + to_csv_timestamp + 'gpt3_tweets.csv'
    db_tweets.to_csv(filename, index=False)

    print('Scraping has completed!')


# keywords to seach by
search_words = "#GPT3 OR #GPT-3 OR #gpt3 OR #gpt-3 OR #Gpt-3 OR #Gpt3 OR GPT3 OR GPT-3 OR gpt3 OR gpt-3"
numTweets = 1500
numRuns = 1
# Scrap tweets
scraptweets(search_words, numTweets, numRuns)
