import praw
import os
import csv
import pandas as pd




username= 'djw0001'
password= 'Abcd354112'
client_id= 'j0TcMKvHvT0hlw'
client_secret= 'TXEowC9Mu_Dw8YJKisSGukBI9PM'

reddit = praw.Reddit(client_id=client_id, client_secret=client_secret,
                    password=password, user_agent='<Python>:<cGP7e8gMlUqQwQ>:<v 1.0>(by /u/djw009)',
                    username=username)


def top_level_comment_gatherer(subreddit, threshhold, filename):
    comment_dataframe = pd.DataFrame(columns = ['comment body', 'comment author', 'comment id', 'subreddit'])
    counter = 0
    for submission in reddit.subreddit(subreddit).hot(limit = 1000):
        if counter < threshhold:
            submission.comments.replace_more(limit = None) #replace "MoreComments" objects
            for comment in submission.comments.list():
                try:
                    body = comment.body
                    author = comment.author
                    id = comment.id
                    comment_dataframe.loc[counter] = [body, author, id, subreddit]
                    counter += 1
                    print(counter)
                except:
                    continue
            if np.mod(counter, 200) == 0:
                comment_dataframe.to_csv(filename, index = False)
        else:

            break

    comment_dataframe.to_csv(filename, index = False)
