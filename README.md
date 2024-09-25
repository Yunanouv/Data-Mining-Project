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

### Step 1 
1. Use a topic model (e.g., PLSA or LDA) to extract topics from all the review text (or a large sample of them) and visualize the topics to understand what people have talked about in these reviews.  
2. Do the same for two subsets of reviews that are interesting to compare (e.g., positive vs. negative reviews for a particular cuisine or restaurant), and visually compare the topics extracted from the two subsets to help understand the similarities and differences between these topics extracted from the two subsets. 


**Answer**  
Here, we're choosing LDA since we have a large dataset in JSON file format which is known for its ability to handle large datasets and produce high-quality topics. LDA is generally considered a more robust and flexible model than PLSA due to its use of Dirichlet priors, which help manage uncertainty and improve topic coherence. 

From all the restaurant's data, we chose 10 samples of topics as shown by the radial dendrogram below. This is the visualization of what people have discussed in restaurant reviews on 10 topics.

<img width="489" alt="image" src="https://github.com/user-attachments/assets/88c113f0-11d3-4bb7-9415-936ccb061088">  

Here's the visualization of the topic clusters. The visualization provides a clear overview of how often each topic appears in the dataset and their relative importance. This can help identify key themes and topics of interest within the text data, guiding further analysis or action based on the prominence of each topic.
<br>
![task1 1-topic clusters](https://github.com/user-attachments/assets/8ed19c46-3440-4c91-849f-2e4f63e0dae3)
<br>

Then, let's take a look at the sample restaurant we have. We chose 3 restaurants for Mexican food to see how the reviews were. They are Filiberto's Mexican Food, Elvira's Mexican Food, and Carolina's Mexican Food. The dendrogram shows the topics that are most frequently reviewed by the customers from each rating, 1 to 5.  
<br>
<img width="410" alt="task1 2-3samplerestaurants" src="https://github.com/user-attachments/assets/a90b458a-5638-4d9a-bd48-f35a94951833">

<br>

