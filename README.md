# Data-Mining-Project
This is an *on-going* data mining project from the University of Illinois Urbana-Champaign from Coursera using [Yelp Dataset]()

## Overview
The goal of the Data Mining Project is to provide us with an opportunity to synthesize the knowledge and skills we’ve learned from the courses and apply them to solve real-world data mining challenges. We will be given a restaurant review data set from Yelp and mine this dataset to discover interesting and useful knowledge to help people make dining decisions, including constructing a cuisine map to help people understand the landscape of different cuisines, mining popular dishes of a given cuisine, recommending (i.e., ranking) restaurants for a given dish, and predicting hygiene of a restaurant.

**Dataset**  
Name: Yelp Dataset  
Contents: Restaurant Review  
  - Business info
  - Review
  - Check-in
  - Tip
  - User
Size: 1.2 GB (40K++ business)

## 1.Topic Extraction 
### Popular Topics
Here, we're choosing the LDA model since we have a large dataset in JSON file format which is known for its ability to handle large datasets and produce high-quality topics. LDA is generally considered a more robust and flexible model than PLSA due to its use of Dirichlet priors, which help manage uncertainty and improve topic coherence. 

From all the restaurant's data, we chose 10 samples of topics as shown by the radial dendrogram below. This is the visualization of what people have discussed in restaurant reviews on 10 topics.

<img width="489" alt="image" src="https://github.com/user-attachments/assets/88c113f0-11d3-4bb7-9415-936ccb061088">  

Here's the visualization of the topic clusters. The visualization provides a clear overview of how often each topic appears in the dataset and their relative importance. This can help identify key themes and topics of interest within the text data, guiding further analysis or action based on the prominence of each topic.
<br>
![task1 1-topic clusters](https://github.com/user-attachments/assets/8ed19c46-3440-4c91-849f-2e4f63e0dae3)
<br>

### Comparing Mexican Food Restaurants  
Then, let's take a look at the sample restaurant we have. We chose 3 restaurants for Mexican food to see how the reviews were. They are Filiberto's Mexican Food, Elvira's Mexican Food, and Carolina's Mexican Food. The dendrogram shows the topics that are most frequently reviewed by the customers from each rating, 1 to 5 using the LDA model after preprocessing the data.  
<br>
<img width="410" alt="task1 2-3samplerestaurants" src="https://github.com/user-attachments/assets/a90b458a-5638-4d9a-bd48-f35a94951833">

<br>

## 2. Cuisine Exploration  
In this step, we will work on mining this data set to discover knowledge about cuisines. In the Yelp data set, businesses are tagged with categories. For example, the category "restaurant" identifies all the restaurants. Specific restaurants are also tagged with cuisines (e.g., "Indian" or "Italian"). This provides an opportunity to aggregate all the information about a particular cuisine and obtain an enriched representation of a cuisine using, for example, review text for all the restaurants of a particular cuisine. Such a representation can then be exploited to assess the similarity between two cuisines, which further enables the clustering of cuisines.  

The goal of this task is to mine the data set to construct a cuisine map to visually understand the landscape of different types of cuisines and their similarities. The cuisine map can help users understand what cuisines are available and their relations, which allows for the discovery of new cuisines, thus facilitating exploration of unfamiliar cuisines.  

### Visualization of Cuisine Map
