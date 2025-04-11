import streamlit as st
import os
import shutil
import time
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import gdown  # Make sure gdown is in your requirements.txt

sns.set_style('darkgrid')

# ------------------------------
# Configuration: File Paths
# ------------------------------

# GitHub-hosted files (files below 25MB)
GITHUB_CSV_PATH = "https://raw.githubusercontent.com/xZANDUBALMx/retinopathy_detection/main/train.csv"
GITHUB_MODEL_PATH = "https://raw.githubusercontent.com/xZANDUBALMx/retinopathy_detection/main/efficientnetb1.keras"

# Google Drive folder that contains larger files (e.g., gaussian_filtered_images folder, retinopathy_model.h5)
# Note: This is the shareable folder link from your Drive.
GOOGLE_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1dSAop854I0FP1Acvxrucn4-_aj9Rlzi_?usp=drive_link"

# To use the Google Drive files in your code (e.g., for loading a large model or images),
# you can either mount your Drive locally or download specific files using gdown. 
# Example: Downloading a file from Google Drive (assuming you know its file ID):
#
#   file_id = "YOUR_FILE_ID"
#   url = f"https://drive.google.com/uc?id={file_id}"
#   output = "retinopathy_model.h5"
#   gdown.download(url, output, quiet=False)
#
# Then you can use 'output' as a local path in your code.

# Working directory for temporary files (adjust this as needed)
WORKING_DIR = "D:/College/Jupyter Lab/Major Project"

# ------------------------------
# Utility functions for plotting
# ------------------------------

def plot_barh(series, title):
    fig, ax = plt.subplots()
    series.plot(kind='barh', ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def display_image(image_path, title="Image"):
    img = cv2.imread(image_path)
    if img is None:
        st.warning(f"Image not found at: {image_path}")
        return
    # Convert BGR (OpenCV default) to RGB for display with matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)

def display_plt(fig):
    st.pyplot(fig)

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------

@st.cache(allow_output_mutation=True)
def load_csv_data(csv_url):
    df = pd.read_csv(csv_url)
    # Map diagnosis to binary and label names.
    diagnosis_dict_binary = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
    diagnosis_dict = {
        0: 'No_DR',
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        4: 'Proliferate_DR',
    }
    df['binary_type'] = df['diagnosis'].map(diagnosis_dict_binary.get)
    df['type'] = df['diagnosis'].map(diagnosis_dict.get)
    return df

@st.cache(allow_output_mutation=True)
def load_image_paths(image_dir):
    # Assumes the image_dir is a local path. If you are using Google Drive,
    # ensure that folder is either mounted locally (e.g., via Google Colab mount)
    # or downloaded to your working directory.
    classlist = os.listdir(image_dir)
    filepaths, labels = [], []
    for klass in classlist:
        classpath = os.path.join(image_dir, klass)
        if os.path.isdir(classpath):
            for f in os.listdir(classpath):
                filepaths.append(os.path.join(classpath, f))
                labels.append(klass)
    df_paths = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
    return df_paths

# ------------------------------
# Data Augmentation Function
# ------------------------------

def run_augmentation(df, working_dir, target=1500):
    aug_dir = os.path.join(working_dir, 'aug')
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    
    # Create class folders in aug_dir.
    for label in df['labels'].unique():
        os.mkdir(os.path.join(aug_dir, label))
    
    # Define augmentation generator.
    gen = ImageDataGenerator(
        horizontal_flip=True,  
        rotation_range=20, 
        width_shift_range=0.2,
        height_shift_range=0.2, 
        zoom_range=0.2
    )
    
    groups = df.groupby('labels')
    for label in df['labels'].unique():
        group = groups.get_group(label)
        sample_count = len(group)
        if sample_count < target:
            aug_img_count = 0
            delta = target - sample_count
            target_dir = os.path.join(aug_dir, label)
            aug_gen = gen.flow_from_dataframe(
                group,  
                x_col='filepaths',
                y_col=None,
                target_size=(224, 224),
                class_mode=None,
                batch_size=1,
                shuffle=False,
                save_to_dir=target_dir,
                save_prefix='aug-',
                save_format='jpg'
            )
            while aug_img_count < delta:
                images = next(aug_gen)
                aug_img_count += len(images)
                
    aug_fpaths, aug_labels = [], []
    for label in os.listdir(aug_dir):
        class_path = os.path.join(aug_dir, label)
        for f in os.listdir(class_path):
            aug_fpaths.append(os.path.join(class_path, f))
            aug_labels.append(label)
    aug_df = pd.DataFrame({'filepaths': aug_fpaths, 'labels': aug_labels})
    
    ndf = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
    return df, aug_df, ndf, aug_dir

# ------------------------------
# Model and Training Functions
# ------------------------------

def build_model(img_shape=(224,224,3), num_classes=5):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=img_shape),
        layers.Conv2D(8, (3,3), padding="valid", activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.BatchNormalization(),

        layers.Conv2D(16, (3,3), padding="valid", activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.BatchNormalization(),

        layers.Conv2D(32, (4,4), padding="valid", activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_history(history):
    # Plot accuracy
    fig1, ax1 = plt.subplots()
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    st.pyplot(fig1)

    # Plot loss
    fig2, ax2 = plt.subplots()
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    st.pyplot(fig2)

# ------------------------------
# Evaluation Function
# ------------------------------

def evaluate_model(test_gen, preds):
    class_dict = test_gen.class_indices
    inv_class_dict = {v: k for k, v in class_dict.items()}
    y_true = test_gen.labels
    y_pred = [np.argmax(p) for p in preds]
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    
    report = classification_report(y_true, y_pred, target_names=list(inv_class_dict.values()))
    st.text_area("Classification Report", report, height=300)

# ------------------------------
# Main Streamlit App
# ------------------------------

def main():
    st.title("Diabetic Retinopathy Classification Web App")
    st.write("This app demonstrates data exploration, augmentation, training, and evaluation using TensorFlow/Keras with files hosted on GitHub and Google Drive.")

    # Sidebar: Set file paths
    st.sidebar.header("Configuration")
    # GitHub file paths (for smaller files)
    csv_path = st.sidebar.text_input("CSV file path (GitHub)", GITHUB_CSV_PATH)
    model_path = st.sidebar.text_input("EfficientNetB1 Model (GitHub)", GITHUB_MODEL_PATH)
    
    # Google Drive paths (for larger files)
    # Here the expectation is that you have a local mount or have downloaded the files via gdown.
    image_dir = st.sidebar.text_input("Gaussian Filtered Images Dir (Google Drive)", os.path.join(WORKING_DIR, "gaussian_filtered_images"))
    # If using a large model from Google Drive, you could download it using gdown (uncomment and set file_id accordingly)
    # large_model_file_id = "YOUR_LARGE_MODEL_FILE_ID"
    # large_model_url = f"https://drive.google.com/uc?id={large_model_file_id}"
    # large_model_path_local = os.path.join(WORKING_DIR, "retinopathy_model.h5")
    # gdown.download(large_model_url, large_model_path_local, quiet=False)
    large_model_path = st.sidebar.text_input("Retinopathy Model (Google Drive)", os.path.join(WORKING_DIR, "retinopathy_model.h5"))
    
    working_dir = st.sidebar.text_input("Working Directory", WORKING_DIR)
    
    activity = st.sidebar.radio("Choose Activity", ("Data Exploration", "Augmentation Preview", "Train Model", "Evaluate Model"))
    
    if activity == "Data Exploration":
        st.header("Data Exploration")
        try:
            df_csv = load_csv_data(csv_path)
            st.subheader("CSV Data Head")
            st.dataframe(df_csv.head())
            st.subheader("Type Value Counts")
            plot_barh(df_csv['type'].value_counts(), "Diagnosis Type Counts")
            st.subheader("Binary Type Value Counts")
            plot_barh(df_csv['binary_type'].value_counts(), "Binary Diagnosis Counts")
        except Exception as e:
            st.error(f"Error loading CSV data: {e}")
    
    elif activity == "Augmentation Preview":
        st.header("Augmentation Preview")
        if os.path.exists(image_dir):
            df_paths = load_image_paths(image_dir)
            st.write("Original File Paths DataFrame (first 5 rows):")
            st.dataframe(df_paths.head())
            
            # Sample images for display
            sample_imgs = {
                "Mild": os.path.join(image_dir, "Mild", "0124dffecf29.png"),
                "Moderate": os.path.join(image_dir, "Moderate", "0161338f53cc.png"),
                "No_DR": os.path.join(image_dir, "No_DR", "00cc2b75cddd.png"),
                "Proliferate_DR": os.path.join(image_dir, "Proliferate_DR", "034cb07a550f.png"),
                "Severe": os.path.join(image_dir, "Severe", "042470a92154.png")
            }
            for key, path in sample_imgs.items():
                if os.path.exists(path):
                    st.subheader(f"Sample: {key}")
                    display_image(path, key)
                else:
                    st.warning(f"Image for {key} not found at {path}")
            
            st.subheader("Running Augmentation")
            try:
                df_original, aug_df, ndf, aug_dir = run_augmentation(df_paths, working_dir)
                st.write("Original counts:")
                st.write(df_original['labels'].value_counts())
                st.write("Augmented counts:")
                st.write(aug_df['labels'].value_counts())
                st.write("Combined counts:")
                st.write(ndf['labels'].value_counts())
                st.write("Augmentation folder structure:")
                for klass in os.listdir(aug_dir):
                    class_path = os.path.join(aug_dir, klass)
                    count = len(os.listdir(class_path))
                    st.write(f"Class: {klass} - {count} images")
            except Exception as e:
                st.error(f"Error running augmentation: {e}")
        else:
            st.error("Image directory not found. Please check the path.")
    
    elif activity == "Train Model":
        st.header("Model Training")
        if os.path.exists(image_dir):
            df_paths = load_image_paths(image_dir)
            _, _, ndf, _ = run_augmentation(df_paths, working_dir)
            
            train_split = 0.8
            valid_split = 0.1
            dummy_split = valid_split / (1 - train_split)
            train_df, dummy_df = train_test_split(ndf, train_size=train_split, shuffle=True, random_state=123)
            valid_df, test_df = train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=123)
            st.write("Train, Validation, and Test split counts:")
            st.write("Train:", len(train_df), "Validation:", len(valid_df), "Test:", len(test_df))
            
            img_size = (224, 224)
            batch_size = 40
            trgen = ImageDataGenerator(preprocessing_function=lambda x: x, horizontal_flip=True)
            tvgen = ImageDataGenerator(preprocessing_function=lambda x: x)
            
            train_gen = trgen.flow_from_dataframe(
                train_df, x_col='filepaths', y_col='labels', 
                target_size=img_size, class_mode='categorical',
                color_mode='rgb', shuffle=True, batch_size=batch_size
            )
            valid_gen = tvgen.flow_from_dataframe(
                valid_df, x_col='filepaths', y_col='labels', 
                target_size=img_size, class_mode='categorical',
                color_mode='rgb', shuffle=True, batch_size=batch_size
            )
            
            model = build_model(img_shape=(224,224,3), num_classes=5)
            st.write("Model Summary:")
            summary_str = []
            model.summary(print_fn=lambda x: summary_str.append(x))
            st.text("\n".join(summary_str))
            
            epochs = st.number_input("Number of Epochs", min_value=1, max_value=100, value=30, step=1)
            st.info("Training started. This might take a while...")
            history = model.fit(
                train_gen,
                epochs=epochs,
                validation_data=valid_gen,
                verbose=1
            )
            st.success("Training complete!")
            plot_history(history)
            
            save_option = st.checkbox("Save trained model", value=False)
            if save_option:
                model_save_path = os.path.join(working_dir, "trained_model.keras")
                model.save(model_save_path)
                st.write(f"Model saved at: {model_save_path}")
        else:
            st.error("Image directory not found. Please check the path.")
    
    elif activity == "Evaluate Model":
        st.header("Model Evaluation")
        # Load the efficientnetb1.keras model from GitHub.
        try:
            model = tf.keras.models.load_model(model_path)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model from GitHub path: {e}")
            return
        
        if os.path.exists(image_dir):
            df_paths = load_image_paths(image_dir)
            _, _, ndf, _ = run_augmentation(df_paths, working_dir)
            train_split = 0.8
            valid_split = 0.1
            dummy_split = valid_split / (1 - train_split)
            train_df, dummy_df = train_test_split(ndf, train_size=train_split, shuffle=True, random_state=123)
            valid_df, test_df = train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=123)
            
            img_size = (224, 224)
            tvgen = ImageDataGenerator(preprocessing_function=lambda x: x)
            length = len(test_df)
            possible_batches = [int(length/n) for n in range(1, length+1) if length % n == 0 and (length/n) <= 80]
            test_batch_size = sorted(possible_batches, reverse=True)[0] if possible_batches else 40
            test_gen = tvgen.flow_from_dataframe(
                test_df, x_col='filepaths', y_col='labels', 
                target_size=img_size, class_mode='categorical',
                color_mode='rgb', shuffle=False, batch_size=test_batch_size
            )
            st.info("Running predictions on the test set...")
            preds = model.predict(test_gen, steps=int(len(test_df)/test_batch_size)+1)
            evaluate_model(test_gen, preds)
        else:
            st.error("Image directory not found. Please check the path.")

if __name__ == '__main__':
    main()
