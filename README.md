# Movie-Success-Prediction
🎬 Movie Success Predictor
A machine learning project that predicts a movie’s box office collection based on key features like budget, runtime, IMDb rating, and even script content using NLP techniques.

🔍 Overview
This project uses historical metadata and textual analysis of movie scripts to forecast the expected revenue and categorize the film as a Hit or Flop. It blends numerical features with text-based inputs to build a robust regression model.

🧠 Technologies Used
Python

Pandas, NumPy

Scikit-learn

TF-IDF (Text feature extraction)

NLTK (text preprocessing)

Matplotlib (for visualizations)

📂 Features
Accepts both CSV-based batch predictions and custom user input via command line.

Automatically cleans and processes raw movie scripts.

Predicts box office revenue and prints whether the movie is a HIT or FLOP.

Includes sample dataset and pre-loaded scripts for testing.

EXAMPLE:

🎬 Custom Mode: Enter your movie details below:
Enter movie title: A Known Stranger
Enter budget (in USD): 10000000
Enter runtime (in minutes): 120
Enter IMDb rating (0-10): 7
Paste movie script or summary:
two best friends became lovers

🎯 Predicted Box Office Collection: $540,380,000.00  
THE MOVIE IS PREDICTED AS HIT
