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

sns.set_style('darkgrid')

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
    # Convert BGR (cv2 default) to RGB for matplotlib
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
def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    # Map diagnosis to binary and label names.
    diagnosis_dict_binary = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
    diagnosis_dict = {
        0: 'No_DR',
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
        4: 'Proliferate_DR',
    }
    df['binary_type'] =  df['diagnosis'].map(diagnosis_dict_binary.get)
    df['type'] = df['diagnosis'].map(diagnosis_dict.get)
    return df

@st.cache(allow_output_mutation=True)
def load_image_paths(image_dir):
    # Assuming image_dir contains subdirectories with class names.
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
    """
    Create augmentation folder structure, run augmentation using ImageDataGenerator,
    and return a dataframe with original + augmented image file paths.
    """
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
    # Augment classes with less than target images.
    for label in df['labels'].unique():
        group = groups.get_group(label)
        sample_count = len(group)
        if sample_count < target: 
            aug_img_count = 0
            delta = target - sample_count  
            target_dir = os.path.join(aug_dir, label)
            # Use flow_from_dataframe; note that y_col is None here.
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
    # Now, read augmented file paths
    aug_fpaths, aug_labels = [], []
    for label in os.listdir(aug_dir):
        class_path = os.path.join(aug_dir, label)
        for f in os.listdir(class_path):
            aug_fpaths.append(os.path.join(class_path, f))
            aug_labels.append(label)
    aug_df = pd.DataFrame({'filepaths': aug_fpaths, 'labels': aug_labels})
    # Concatenate original and augmented dataframes.
    ndf = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
    return df, aug_df, ndf, aug_dir

# ------------------------------
# Model and Training Functions
# ------------------------------

def build_model(img_shape=(224,224,3), num_classes=5):
    # Example: Using a custom small CNN model for simplicity.
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
    # Plot accuracy.
    fig1, ax1 = plt.subplots()
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    st.pyplot(fig1)

    # Plot loss.
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
    
    # Confusion Matrix.
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    
    # Classification Report.
    report = classification_report(y_true, y_pred, target_names=list(inv_class_dict.values()))
    st.text_area("Classification Report", report, height=300)

# ------------------------------
# Main Streamlit App
# ------------------------------

def main():
    st.title("Diabetic Retinopathy Classification Web App")
    st.write("This application demonstrates data exploration, augmentation, training, and evaluation of a diabetic retinopathy classification model using TensorFlow/Keras on a Streamlit web interface.")
    
    # Sidebar: Set file paths (update these paths as needed)
    st.sidebar.header("Configuration")
    csv_path = st.sidebar.text_input("CSV file path", "D:/College/Jupyter Lab/Major Project/train.csv")
    image_dir = st.sidebar.text_input("Gaussian filtered images dir", "D:/College/Jupyter Lab/Major Project/gaussian_filtered_images")
    working_dir = st.sidebar.text_input("Working directory", "D:/College/Jupyter Lab/Major Project")
    
    activity = st.sidebar.radio("Choose Activity", 
                                ("Data Exploration", "Augmentation Preview", "Train Model", "Evaluate Model"))
    
    if activity == "Data Exploration":
        st.header("Data Exploration")
        if os.path.exists(csv_path):
            df_csv = load_csv_data(csv_path)
            st.subheader("CSV Data Head")
            st.dataframe(df_csv.head())
            st.subheader("Type Value Counts")
            plot_barh(df_csv['type'].value_counts(), "Diagnosis Type Counts")
            st.subheader("Binary Type Value Counts")
            plot_barh(df_csv['binary_type'].value_counts(), "Binary Diagnosis Counts")
        else:
            st.error("CSV path not found. Please check the path.")
    
    elif activity == "Augmentation Preview":
        st.header("Augmentation Preview")
        if os.path.exists(image_dir):
            df_paths = load_image_paths(image_dir)
            st.write("Original File Paths Dataframe (first 5 rows):")
            st.dataframe(df_paths.head())
            
            # Display sample images from each class (using your provided image examples)
            # You can adjust these image paths as needed
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
            
            # Run augmentation and show the number of images per class.
            st.subheader("Running Augmentation")
            df_original, aug_df, ndf, aug_dir = run_augmentation(df_paths, working_dir)
            st.write("Original counts:")
            st.write(df_original['labels'].value_counts())
            st.write("Augmented counts:")
            st.write(aug_df['labels'].value_counts())
            st.write("Combined counts:")
            st.write(ndf['labels'].value_counts())
            
            # List the augmentation directory contents.
            st.write("Augmentation folder structure:")
            for klass in os.listdir(aug_dir):
                class_path = os.path.join(aug_dir, klass)
                count = len(os.listdir(class_path))
                st.write(f"Class: {klass} - {count} images")
        else:
            st.error("Image directory not found. Please check the path.")
    
    elif activity == "Train Model":
        st.header("Model Training")
        # Here we assume that you have created your augmented (or combined) dataframe
        if os.path.exists(image_dir):
            df_paths = load_image_paths(image_dir)
            # For demonstration, we run augmentation.
            _, _, ndf, _ = run_augmentation(df_paths, working_dir)
            
            # Split the data.
            train_split = 0.8
            valid_split = 0.1
            dummy_split = valid_split / (1 - train_split)
            train_df, dummy_df = train_test_split(ndf, train_size=train_split, shuffle=True, random_state=123)
            valid_df, test_df = train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=123)
            st.write("Train, Validation, and Test split counts:")
            st.write("Train:", len(train_df), "Validation:", len(valid_df), "Test:", len(test_df))
            
            # Create ImageDataGenerators.
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
            
            # Build the model.
            model = build_model(img_shape=(224,224,3), num_classes=5)
            st.write("Model Summary:")
            # Capture model.summary() into a string.
            summary_str = []
            model.summary(print_fn=lambda x: summary_str.append(x))
            st.text("\n".join(summary_str))
            
            # Run training.
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
            
            # Save model if desired.
            save_option = st.checkbox("Save trained model", value=False)
            if save_option:
                model_save_path = os.path.join(working_dir, "trained_model.keras")
                model.save(model_save_path)
                st.write(f"Model saved at: {model_save_path}")
        else:
            st.error("Image directory not found. Please check the path.")
    
    elif activity == "Evaluate Model":
        st.header("Model Evaluation")
        # For evaluation we assume that the model is already trained and saved. 
        # You can load the model from GitHub or local storage.
        model_path = st.sidebar.text_input("Trained Keras Model Path", "D:/College/Jupyter Lab/Major Project/efficientnetb1.keras")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            st.success("Model loaded successfully!")
            
            # Recreate data generators for test set.
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
                # Calculate test batch size (ensuring that batch size divides test set size)
                length = len(test_df)
                possible_batches = [int(length/n) for n in range(1, length+1) if length % n == 0 and (length/n) <= 80]
                if possible_batches:
                    test_batch_size = sorted(possible_batches, reverse=True)[0]
                else:
                    test_batch_size = 40
                test_gen = tvgen.flow_from_dataframe(
                    test_df, x_col='filepaths', y_col='labels', 
                    target_size=img_size, class_mode='categorical',
                    color_mode='rgb', shuffle=False, batch_size=test_batch_size
                )
                # Run predictions.
                st.info("Running predictions on the test set...")
                preds = model.predict(test_gen, steps=int(len(test_df)/test_batch_size)+1)
                evaluate_model(test_gen, preds)
            else:
                st.error("Image directory not found. Please check the path.")
        else:
            st.error("Model path not found. Please check the path.")

if __name__ == '__main__':
    main()
