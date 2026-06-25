# Art Recommender

This project aims to explore the concept of art recommendations.  
Focusing on visual and semantic features and their relationship, hoping to capture the unique characteristics and nuances of artworks.  
The final objective is to build an art discovery tool by using a recommendation system to suggest similar artworks to the ones chosen by the user with the option to adjust importance of visual and semantic features.

The project consists of two main components: the data processing pipeline with recommendation system and the user application. With such structure the initial MET dataset had to be divided and prepared for different purposes: display or similarity model.

## Data layer

The MET dataset was sampled in order to reduce the number of entries from twenty thousand to just two thousand. At an early stage, features from the sample database were separated into two separate DataFrames: `app_df` and `ml_df`.

### I. app_df

The app_df is used to store information for display in the user application. The only processed applied to this DataFrame were simple cleaning and filling in unknow values as “Unknown”. The dataset contains features such as the artwork title, artist, century, country of origin, medium, object type, department, tags, image URL and a link to the official Met gallery page.

### II. ml_df

The ml_df was prepared to store the features needed to develop the recommendation system. It underwent additional preprocessing and feature engineering in order to prepare it for the similarity calculations. In the final stage the DataFrame was divided into `semantic_df` and `visual_df`. The first containing information directly from the MET database and the latter, newly extracted features from the artwork images.

Considering the split in the data, it was important to maintain data integrity and alignment. That is why each table includes a primary key (`object_id`) and each table was made sure to have the same set of object ids.

## Pipeline design

The pipeline was designed as an ETL data pipeline. Its purpose was to transform the raw MET dataset into usable `app_df` and `ml_df` DataFrames. In the extraction stage, useful metadata was collected from the CSV source data, then the data underwent filtering, cleaning and sampling. Only after that the API was connected to reduce the number of calls. Images of artworks were downloaded and visual features extracted.

## Application design

The application was designed to make the art exploration easy and engaging. Four pages were created to naturally follow the exploration process. Starting with the main page, that includes six art categories: Greek and Roman Art, Medieval Art, Asian Art, Islamic Art, European Paintings and Modern and Contemporary Art. The user picks one of these and is taken to the Collection page where a library of images is displayed with the artwork’s title and author. User can choose any of them and open the Single artwork page, where they will find more detailed information about the artwork as well as the section with similar artworks. The recommendation approach (semantic/visual) can be adjusted by the user on a slider.

Finally, on a personal note, I decided to include a Women Artists page to better represent the 0.6% of artworks created by women. If the user discovers a painting by a woman, there will be a pop up with a link to the Women Artists Page. A shortcut to this page was also added in the Home page in the form of a pink star in the corner.
