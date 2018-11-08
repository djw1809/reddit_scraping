import praw
import os
import csv
import pandas as pd
import numpy as np 


#[bot1]
username= 'djw0001'
password= 'Abcd354112'
client_id= 'j0TcMKvHvT0hlw'
client_secret= 'TXEowC9Mu_Dw8YJKisSGukBI9PM'

#[bot2]
username2 = 'djw0002'
password2 = 'Abcd354112'
client_id2 = 'j7E1Qlv881SzyA'
client_secret2 = 'hTI4KMkFumtvOXvM6HDOTsr_MYM'

#[bot4]
username3 = 'djw0003'
password3 = 'Abcd354112'
client_id3 = '6FP8876h655ZkQ'
client_secret3 = 'Fs681Uqqvvhd2cyKCLDqwJeem44'

#[bot5]
username4 = 'djw0004'
password4 = 'Abcd354112'
client_id4 = 'eQkoKOCRV3acNw'
client_secret4 = 'o3dnk3uW_-5rLRFcW0g6rJIWLag'

#[bot6]
username5 = 'djw0005'
password5 = 'Abcd354112'
clint_id5 = 'eey7jTJOeBh_9A'
client_secret5 = '3l7Q5MKMKU6RD9Uq1g3CtGa6-zo'

#[bot7]
username6 = 'djw00005'
password6 = 'Abcd354112'
client_id6 = 'yLkh80ENCNbsgQ'
client_secret6 = '3QR8piuNigrunr8UGIxY6FQLmzg'

#[bot8]
username7 = 'djw0006'
password7 = 'Abcd354112'
client_id7 = 'owytZ5MO5y-ilg'
client_secret7 = '9ooFy80uN-gY15ZE7DXkRBS49HY'

#[bot9]
username8 = 'djw007'
password8 = 'Abcd354112'
client_id8 = '5TwEmJEviGiMiw'
client_secret8= 'SnYsGi9HttHT6L71RYNhXJf8P90'




def top_level_comment_gatherer(subreddit, threshhold, filename, bot):

    if bot == 1:
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret,
                            password=password, user_agent='praw_comment_scraper:<v 1.0>(by djw009)',
                            username=username)
    elif bot == 2:
        reddit = praw.Reddit(client_id=client_id2, client_secret=client_secret2,
                            password=password2, user_agent='praw_comment_scraper:<v 1.0>(by djw009)',
                            username=username2)

    elif bot == 3:
        reddit = praw.Reddit(client_id=client_id3, client_secret=client_secret3,
                            password=password3, user_agent='praw_comment_scraper:<v 1.0>(by djw009)',
                            username=username3)

    elif bot == 4:
        reddit = praw.Reddit(client_id=client_id4, client_secret=client_secret4,
                            password=password4, user_agent='praw_comment_scraper:<v 1.0>(by djw009)',
                            username=username4)
    elif bot == 5:
        reddit = praw.Reddit(client_id=client_id5, client_secret=client_secret5,
                            password=password5, user_agent='praw_comment_scraper:<v 1.0>(by djw009)',
                            username=username5)

    elif bot == 6:
        reddit = praw.Reddit(client_id=client_id6, client_secret=client_secret6,
                            password=password, user_agent='praw_comment_scraper:<v 1.0>(by djw009)',
                            username=username)
    elif bot == 7:
        reddit = praw.Reddit(client_id=client_id7, client_secret=client_secret7,
                            password=password7, user_agent='praw_comment_scraper:<v 1.0>(by djw009)',
                            username=username7)
    elif bot == 8:
        reddit = praw.Reddit(client_id=client_id8, client_secret=client_secret8,
                            password=password8, user_agent='praw_comment_scraper:<v 1.0>(by djw009)',
                            username=username8)
    elif bot == 9:
        reddit = praw.Reddit(client_id=client_id9, client_secret=client_secret9,
                            password=password9, user_agent='praw_comment_scraper:<v 1.0>(by djw009)',
                            username=username9)


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

def comment_screener(dataset):
    for i in dataset.index:
        print(dataset.loc[i, 'comment body'])
        keep = input('keep? 1 - yes 0 - no')
        if keep == '1':
            pass
        else:
            dataset = dataset.drop(i)
        if np.mod(i, 10) == 0:
            quit = input('quit? 1 - yes 0 - no')
            if quit == '1':
                break
            
                
    dataset.index = range(len(dataset))
    return dataset
