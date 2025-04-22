# MovieShowRecs

**MovieShowRecs** is a recommendation system designed to suggest similar movies and TV shows based on their descriptions. The system uses TF-IDF vectorization and cosine similarity to find content that is most similar to the one you input.

### Features
- Recommends movies and TV shows separately.
- Uses machine learning techniques like TF-IDF and cosine similarity.
- Built with **Python** and **Streamlit** for easy usage and deployment.
  
### Technologies
- **Python**
- **pandas** - For data manipulation
- **scikit-learn** - For TF-IDF and cosine similarity
- **Streamlit** - For the user interface
- **numpy** - For array operations

### How It Works
1. The app loads data from CSV files containing movie and TV show descriptions.
2. TF-IDF vectorization is applied to convert the text into numerical data.
3. Cosine similarity is used to calculate how similar different items are.
4. The app provides recommendations based on the similarity score.

### Requirements
- Python 3.x
- Install the required packages:
  
  ```bash
  pip install -r requirements.txt
