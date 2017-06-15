# Fake News Challenge 

The idea of fake news is often referred to as click-bait in social trends and is defined as a “made up story with an intention to deceive, geared towards getting clicks”, Tavernise (2016). Some news articles have titles which grab a reader’s interest. Yet, the author only emphasizes a specific part of the article in the title. If the article itself does not focus on or give much truth to what the title had written, the news may be misleading.

The goal of this project is to use natural language processing techniques to automate stance detection, since it is not practical for humans to fact check every piece of information produced by the media. 

Stance detection is a method used to determine the quality of a news article by taking into consideration what other organisations write about the same headline. A body of text is claimed to agree, disagree, discuss, or be unrelated to a headline, Fake News Challenge (2016) Stance detection is the method that will be used to determine the quality of a news source. From the [FakeChallenge.org](http://fakenewschallenge.org) a dataset will be provided which consists of a headline and a body of text. This body of text may be from a different article. Allowing bodies of text from different articles allows this system to take into account what the other organisations are saying about he same headline. The output of the system will be the stance of the body of text related to the title. As shown in the fake news challenge [fake news source] the system will support will support the following stance types:
- Agrees
- Disagrees 
- Discusses
- Unrelated

With this system, for a set of news headlines, statistics can be gathered with respect to the stances. With these statistics, a user can come to their own conclusion of whether a new organisation has reputable news sources. To achieve these stances, this system will train on the data supplied by the fake news challenge. This data will provide the stance along with the headline and body to allow the system to learn which word combinations lead to which stance. For testing, data will be provided without the stances. To expand upon the baseline, this project will consider stemming words, removing stop words, and smoothing.


# Developer commands
To generate the features
python feature_generation.py

To train and get analysis
python svm.py

# Sources
[FNC baseline repo](https://github.com/FakeNewsChallenge/fnc-1-baseline)
[FakeChallenge.org](http://fakenewschallenge.org).
[Fnc-1 repo](https://github.com/FakeNewsChallenge/fnc-1) 




