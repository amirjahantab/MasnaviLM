# MasnaviLM: Building a Word-Level Language Model for Persian

This repository contains the implementation of a word-level language model for the Persian language, specifically focusing on classical Persian literature. The dataset used is *Masnavi* by Rumi, one of the most renowned works in Persian literature. The project is designed to develop a word-level language model using techniques from Natural Language Processing (NLP) and Machine Learning (ML).

## Features
- **Language Modeling:** A word-level language model using TensorFlow/Keras.
- **Data Preprocessing:** Tokenization, vectorization, and padding of text sequences.
- **Training & Validation:** Training the model on Persian text, using cross-entropy loss and Adam optimizer.
- **Text Generation:** Generating sequences of Persian text based on the trained model.

## Project Structure
The key file for this project is the Jupyter notebook:
- [`Building_Language_Model_For_Persian-Word-Level.ipynb`](Building_Language_Model_For_Persian-Word-Level.ipynb): The main notebook containing the entire pipeline for preprocessing the Persian text, building the word-level language model, and generating text.

### Key Components:
1. **Data Preprocessing**
   - **Text Tokenization:** Converting raw Persian text into word tokens.
   - **Vectorization:** Mapping each unique token to an integer.
   - **Sequence Padding:** Ensuring uniform sequence lengths for batch training.

2. **Model Architecture**
   - **Embedding Layer:** Maps the input tokens to dense vectors.
   - **LSTM Layers:** Long Short-Term Memory (LSTM) layers are used to capture long-range dependencies in text.
   - **Dense Output Layer:** Predicts the next word in the sequence from the set of all possible words.

3. **Training Process**
   - **Loss Function:** Categorical cross-entropy for multi-class classification.
   - **Optimizer:** Adam optimizer is used to minimize the loss function.
   - **Epochs & Batch Size:** Configurable training parameters.

4. **Text Generation**
   - After training, the model can be used to generate Persian text by predicting the next word in a sequence based on a seed input.

