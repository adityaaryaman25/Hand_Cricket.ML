# Hand_Cricket.ML

ML Hand Cricket is a computer vision game that reads webcam hand gestures as player moves. Using machine learning, it analyzes each player’s patterns to predict their next move. Unique player IDs and learning profiles let the AI adapt over time, with adjustable difficulty and an interactive interface.


Overview

This project is an advanced real-time hand gesture recognition and interactive Hand Cricket gaming system built using Python, OpenCV, MediaPipe, NumPy, and Scikit-learn. It detects hand landmarks through a webcam, converts them into numerical features, recognizes gestures representing numbers from 1 to 10, and uses predictive modeling (such as Markov chains, Bayesian models, or ML) to play a classic game of Hand Cricket against the user.FeaturesReal-time Hand Tracking: Detects hand landmarks using MediaPipe.   Custom Gesture Mapping: Associates specific hand configurations with numbers 1–10.Dataset Collection: Captures normalized hand landmark coordinates and saves them in CSV format.Machine Learning Classification: Trains a Random Forest classifier on collected gesture datasets.Live Prediction & Smoothing: Recognizes gestures via webcam with confidence filtering and smoothing windows to stabilize frame-to-frame results.Interactive Hand Cricket AI: Connects real-time gesture recognition with an intelligent game engine that anticipates player moves using historical data and predictive modeling.

Instructions:
1). Import the packages/modules from requirements.txt
2). We have trained the models on our data so the use just has to run "furnish_hand_cricket_ai.py" either by command or using the IDE run button.
3). Please follow the gestures/options guidelines as given in the submission.
(Brief guidelines for users ... i. play the AI like a normal game of hand cricket.
                                ii. Numbers 1,2,3,4,5 are normal gestures, use your right hand and make sure for nums (1-5) you palm faces the camera.
                                iii. Numbers 6,7,8,9 are gestures followed up after the normal 1-5 numbers and your palm should face away from the camera.
                                iv. Number 10 is the gesture of 0 with palm facing towards the camera.
                                V. MOST IMPORTANT: *** PUT YOUR HAND UP ONE WHEN IT PROMPTS YOU TO, i.e., WHEN THE FRAME TURNS RED. ***




Project Structure:

SinTaxError2026/
│
├── .venv/
├── data/
├── hand_cricket_ai/
├── models/
│
├── furnish_hand_cricket_ai.py
├── gesture_features.py
├── hand_cricket_matches.csv
├── hand_cricket_moves.csv
├── hand_cricket_profile.json
├── hand_detector.py
├── step3_define_gestures.py
├── step4_collect_data.py
├── step5_train_model.py
├── step6_live_prediction.py
├── requirements.txt
└── README.md

File DescriptionsC
ore Pipeline Scripts
step3_define_gestures.py: Allows users to define and map custom hand gestures for numbers 1 to 10, saving descriptions to data/gesture_mapping.json.
step4_collect_data.py: Captures snapshots of finger positions and normalized landmark coordinates to build the training dataset (data/gestures.csv).
step5_train_model.py: Trains a Random Forest machine learning classifier using the collected gesture dataset and serializes the model.
step6_live_prediction.py: Tests the trained model in real time using webcam input to verify correct gesture recognition.Hand Cricket AI & Support 

Modules
furnish_hand_cricket_ai.py: The primary executable script that bridges real-time hand gesture recognition with the Hand Cricket game engine.
gesture_features.py: Extracts and normalizes the 63 numerical landmark features (21 landmarks $\times$ 3 coordinates) from hand tracking data.
hand_detector.py: Handles webcam initialization and MediaPipe hand-landmark extraction routines.
hand_cricket_matches.csv & hand_cricket_moves.csv: Record historical match data and individual player moves used by the AI to learn patterns.
hand_cricket_profile.json: Stores user profiles, preferences, and tracking states for the Hand Cricket game.


FOLDER INFO:
.venv/: Contains the Python virtual environment files, including installed packages and dependencies isolated for this project.
.data/: Stores CSV files and JSON configurations (such as gesture mappings and datasets) used for training and running the recognition model.
.hand_cricket_ai/: Houses the core logic, sub-modules, and predictive engine components (utilizing approaches like Markov chains, Bayesian models, or machine learning) that power the Hand Cricket game.models/: Stores machine learning assets and binary files, including the MediaPipe landmark task file (hand_landmarker.task) and trained classifier artifacts (like gesture_model.joblib).   
