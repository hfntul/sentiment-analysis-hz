# Sentiment Analysis

**Note: broken english, newbie**

Okay, actually I have so much time lately and days ago when I scrolled through my twitter timeline, I saw a lot of Zemo slander there. I don't know it's like suddenly a bunch of people threw hatred at him. As a Zemo apologist I was kinda pissed seeing that. And then I was beginning to think, did people on twitter hate Zemo during The Falcon and the Winter Soldier aired too? Because you know there's no such hatred at him on my twitter timeline during that time. And now I am curious. Was he actually hated? or loved? So I have an idea to know what actually happened at that time by doing sentiment analysis to determine whether Zemo was hated or loved by MCU stan twitter during The Falcon and the Winter Soldier aired.

## Data Collecting

The data was collected from twitter data using `snscrape` library on python. It contains IDs, user, date, the content of tweet, language, retweet count, like count and quote count. I managed to retrieve 110.034 tweets (all languages) by using `zemo` as the keyword and only fetching the tweets from 18 March, a day before TFATWS aired, until 08 May which was the time marvel released The Making of The Falcon and the Winter Soldier. And by the way, 110.034 is a lot of tweets but it needs to be cleaned first because it still contains so much incorrect, corrupted, incorrectly formatted, duplicate, or incomplete data.

![Screenshot from 2021-09-17 12-39-40](https://user-images.githubusercontent.com/60166588/133730375-fe4b9c1e-8dc4-4dfe-8f6a-674e51460e55.png)

## Data Pre-processing

Before doing the sentiment analysis, first I am gonna show the statistics of tweets that include `zemo` during TFATWS aired per week. To do this, in the data pre-processing I only cleaned the URL, emojis and @ mention first and converted all tweets to lowercase. And then I picked only the tweets that contain `zemo`. I did this to avoid the tweets that contain his name in the mention instead in the tweet. The remaining dataset contains 108.006 tweets now. According to this data, this is the accumulation of tweets per week

![tweet per week](https://user-images.githubusercontent.com/60166588/133727583-cf68e905-73d8-49ea-a7ba-e0d5a5a6c9cb.png)

Nah of course it peaked in the third week, the week when episode 3 aired and when Marvel released the Zemo cut and upload that one hour dancing Zemo on youtube. It's interesting how the stats in the sixth week beat the stats in the fifth week when Zemo only appeared for 15 seconds. King.

And this is the most active users during that time. These people were whipped. The highest one was like tweeting about Zemo ten times a day 😭

![Screenshot from 2021-09-17 11-58-19_2](https://user-images.githubusercontent.com/60166588/133761107-8def0df0-1f3d-4b85-8df2-da7453f15421.png)

Also the most liked tweets. 

![InShot_20210917_124129484_2](https://user-images.githubusercontent.com/60166588/133731585-e955b553-4ca3-43a4-b0cb-a47569537153.jpg)

I also tried to make a word cloud. It’s basically graphical representations of word frequency that give greater prominence to words that appear more frequently in a source text. First I cleaned the data again. This time i removed the hashtag, the non-english tweets and removed the duplicate tweets. The data now is only 63.267 tweets. And here’s the result. I saw `thunderbolt` :)

![cloudword](https://user-images.githubusercontent.com/60166588/133727796-194fff6d-dd39-4b29-82e8-84fd52220ef5.png)

I made another version too using Zemo wearing mask pic but turned out it looks like a thief idk it’s just not like him at all lmao

![zemo cw](https://user-images.githubusercontent.com/60166588/133727792-7f57641a-8ae9-48f9-9f67-95f12f529774.png)

For the last cleaning, I decided to only retrieve max 30 tweets per user to avoid ‘bias’. Those 30 tweets were chosen based on the tweets that have the most likes. So the remaining dataset contains 57.718 tweets now. 

# Sentiment Analysis

Sentiment classification is carried out based on the compound value. Each tweet will have its compound value later that will determine whether that tweet has positive, neutral or negative sentiment. To do this I used vaderSentiment library on python. This is its compound score metric

| Sentiment     | Compound Score |
| ------------- | ------------- |
| Positive     |    ≥ 0.05 |
| Neutral  | > -0.05 and < 0.05 |
| Negative  |    ≤ -0.5 |

This figure below displays `zemo` tweets sentiment distribution. More precisely, there are 26.124 positive sentiment tweets, 16.978 negative sentiment tweets and 14.616 neutral sentiment tweets.

![Screenshot from 2021-09-17 12-17-29](https://user-images.githubusercontent.com/60166588/133728510-6cd710be-4201-467c-b46b-0b97bf94f10d.png)

The histogram on the left shows the compound score for all tweets. And the histogram on the right shows the compound score for 5000 sample tweets by eliminating the tweets that have 0 compound score first. The highest bar is around 0.5 - 0.75 which is the positive sentiment

![Screenshot from 2021-09-17 12-20-52](https://user-images.githubusercontent.com/60166588/133728802-2ec173fa-f251-499a-869e-b03a0b039201.png)

Okay these are some example of positive, neutral and negative sentiment tweets 

Positive (generating the tweets that have compound score ≥ 0.9)
![Screenshot from 2021-09-17 14-46-49](https://user-images.githubusercontent.com/60166588/133745143-e067b4fc-fa3e-4bf7-9c60-81898dcca8da.png)

Neutral (generating the tweets that have compound score = 0)
![Screenshot from 2021-09-17 14-47-04](https://user-images.githubusercontent.com/60166588/133745152-7b8ef385-00ec-472c-adfb-9912e28d7e7d.png)

Negative (generating the tweets that have compound score ≤ -0.9)
![Screenshot from 2021-09-17 14-51-45](https://user-images.githubusercontent.com/60166588/133745753-87b235e8-10ce-4d9d-a950-d0430ccdec49.png)

And these are the word cloud for both positive sentiment tweets and negative sentiment tweets.

![Screenshot from 2021-09-17 15-30-38](https://user-images.githubusercontent.com/60166588/133751408-751dc3de-130f-4c9c-8c75-70e4222def8b.png)

# Evaluating the Model

Based on all those statistics, there are more `zemo` tweets that have positive sentiment than negative sentiment. So what does it mean then? Yeah for a while we can say that he’s more loved than hated by MCU stan twitter during TFATWS aired. Just for a while because we need to evaluate the performance of the model that has been made. Some sort of figuring out how accurate the model is. It can be done by finding the evaluation metrics which is an integral component of any data science project. I used the Multinomial Naive Bayes classification algorithm to evaluate the model and here is the result

![Screenshot from 2021-09-17 14-46-33](https://user-images.githubusercontent.com/60166588/133745780-b5ab40ea-d4d3-45d5-a81f-19fc91e0e7d2.png)

Kinda disappointed because the accuracy is only 68.11% which is so bad I think. But since I like Zemo I am gonna explain that this result actually is pretty good, I know I am biased here hehng. The result is not that bad because the positive sentiment has the highest f1-score and the negative sentiment is the lowest one among those three. And also the positive sentiment's recall score is nice. Recall is defined as the percentage of total relevant results correctly classified by this model. It means all those tweets that have been labeled as a positive sentiment are 83.91% correct. The heatmap below shows it more clearly. 0 = negative, 1 = neutral, 2 = positive

![Screenshot from 2021-09-17 14-54-15](https://user-images.githubusercontent.com/60166588/133746094-d1469d3c-72f8-49e3-9113-4004d32dfd9c.png)

So the final conclusion? Same as before the evaluating process, I can say that, based on the model that has been made, Zemo was more loved than hated by MCU stan twitter during TFATWS aired. Note than sentiment analysis is actually a tricky subject because understanding emotions through text is not always easy so maybe what I am doing is not that accurate.

## References
- [Twitter Sentiment Analysis](https://towardsdatascience.com/step-by-step-twitter-sentiment-analysis-in-python-d6f650ade58d)
- [Using snscrape Library](https://medium.com/swlh/how-to-scrape-tweets-by-location-in-python-using-snscrape-8c870fa6ec25)
- [Vader Sentimen](https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f)
- [Multinomial Naive Bayes](https://towardsdatascience.com/sentiment-analysis-of-tweets-using-multinomial-naive-bayes-1009ed24276b)
