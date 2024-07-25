# Token_ClassificationBERT

In this project I built an NER(Named Entity Recognition ) system using BERT for token classification. Preprocessed data with Pandas and split datasets with Scikit-learn. Trained and fine-tuned the model with Hugging Face Transformers, achieving high accuracy. Evaluated performance using precision, recall, and F1-score. Deployed the model for real-time use and managed it with Hugging Face Hub.

Below I have provided the detailed workflow of the program.

## 1. **Purpose of the Program**

The program aims to train a Named Entity Recognition (NER) model using BERT for token classification. It processes a dataset of sentences and their corresponding NER tags, trains a model, evaluates its performance, and provides functionality for model inference.

## 2. **Core Features**

- **Data Ingestion and Preprocessing**: Loading and processing a CSV dataset containing sentences and NER tags.
- **Tag Mapping**: Creating a mapping of NER tags to unique IDs.
- **Data Splitting**: Dividing the dataset into training and testing sets.
- **Model Training**: Using Hugging Face Transformers library to train a BERT-based NER model.
- **Evaluation**: Computing performance metrics for the trained model.
- **Model Inference**: Providing a pipeline for classifying tokens in new text inputs.
- **Data Saving**: Saving processed data and tag mappings for future use.

## 3. **Technologies Used**

- **Programming Languages**: Python.
- **Libraries and Frameworks**:
  - **Pandas**: For data manipulation.
  - **Scikit-learn**: For data splitting.
  - **Hugging Face Transformers**: For model training, tokenization, and inference.
  - **Evaluate**: For computing evaluation metrics.
- **Data Storage**: Parquet files for dataset storage.
- **Model Management**: Hugging Face Hub for model storage and deployment.

## 4. **Input and Output**

- **Input**: 
  - Dataset in CSV format ("ner.csv") containing sentences and corresponding NER tags.
  - Pre-trained BERT model checkpoint ("bert-base-uncased").
- **Output**: 
  - Trained NER model.
  - Evaluation metrics (precision, recall, F1-score, accuracy).
  - Parquet files ("train_ner.parquet", "test_ner.parquet") for processed training and testing data.
  - JSON file ("tags.json") mapping NER tags to IDs.

## 5. **Major Components**

1. **Data Preparation**
   - **Loading Data**: Reading the CSV file into a Pandas DataFrame.
   - **Tag Mapping**: Creating a dictionary to map unique NER tags to integer IDs.
   - **Data Transformation**: Converting string-formatted tags to lists and mapping them to IDs.
   - **Tokenization**: Splitting sentences into tokens.
   - **Data Splitting**: Using `train_test_split` to divide data into training and testing sets.
   - **Data Saving**: Saving the processed datasets to Parquet files and tag mappings to a JSON file.

2. **NERDataset Class**
   - **Initialization**: Setting paths for training and testing data, and initializing the tokenizer.
   - **Data Loading**: Loading the Parquet files using Hugging Face's `load_dataset`.
   - **Label Alignment**: Aligning NER labels with tokenized inputs.
   - **Preprocessing**: Tokenizing sentences and aligning labels with tokens for training.

3. **NERTrainer Class**
   - **Initialization**: Loading tag mappings, datasets, model, and tokenizer.
   - **Metric Computation**: Using the "seqeval" library to compute evaluation metrics.
   - **Training Arguments**: Defining parameters for training such as output directory, learning rate, epochs, etc.
   - **Model Training**: Training the model using Hugging Face's `Trainer` class and saving the model.

4. **Model Inference**
   - **Pipeline Setup**: Creating a Hugging Face pipeline for token classification.
   - **Inference**: Running the pipeline on new text inputs to perform NER.

## 6. **Workflow**

1. **Data Preparation**
   - Load the dataset from "ner.csv" using `pd.read_csv`.
   - Create a tag-to-ID mapping and save it to "tags.json".
   - Convert string-formatted tags to lists using `literal_eval`.
   - Map tags to IDs and split sentences into tokens.
   - Split the dataset into training and testing sets using `train_test_split`.
   - Save the processed data as "train_ner.parquet" and "test_ner.parquet".

2. **NERDataset Class**
   - Initialize with dataset paths and tokenizer checkpoint.
   - Load training and testing datasets using `load_dataset`.
   - Align word-level labels with tokenized inputs using `align_labels_with_tokens`.
   - Preprocess the data by tokenizing and aligning labels.

3. **NERTrainer Class**
   - Initialize by loading NER labels, datasets, model, and tokenizer.
   - Define a method to compute evaluation metrics with the `evaluate` library.
   - Set training arguments using `TrainingArguments`.
   - Train the model with `Trainer` and save it to Hugging Face Hub.

4. **Model Inference**
   - Set up a Hugging Face pipeline for token classification.
   - Use the trained model checkpoint for inference on new text inputs.

## 7. **Special Algorithms or Techniques**

- **BERT Model**: Leverages the pre-trained "bert-base-uncased" model for token classification.
- **Token Alignment**: Custom method to align NER labels with tokenized inputs, handling sub-tokens generated by BERT.
- **Evaluation Metrics**: Uses "seqeval" for precise NER evaluation metrics.


