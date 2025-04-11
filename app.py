import streamlit as st
import os, re, shutil, time, random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import gdown  # used for downloading files from Google Drive

sns.set_style('darkgrid')

# ---------------------------
# Utility Functions for Google Drive Links
# ---------------------------
def convert_gdrive_url(shared_url):
    """
    Converts a Google Drive shared file URL into a direct downloadable URL.
    For example, if the shared URL is:
        https://drive.google.com/file/d/FILE_ID/view?usp=drive_link
    This returns:
        https://drive.google.com/uc?export=download&id=FILE_ID
    """
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', shared_url)
    if match:
        file_id = match.group(1)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        return download_url
    else:
        return shared_url

@st.cache_data
def download_file(url, output_path):
    """
    Download a file from a direct download URL if it doesn't exist locally.
    """
    if not os.path.exists(output_path):
        st.info(f"Downloading file to {output_path}...")
        gdown.download(url, output_path, quiet=False)
    return output_path

# ---------------------------
# Functions for Data Loading and Preprocessing
# ---------------------------
@st.cache_data
def load_csv(csv_path):
    """Load CSV data (e.g., train.csv) into a DataFrame and map diagnosis."""
    df = pd.read_csv(csv_path)
    # Mapping dictionaries
    diagnosis_dict_binary = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
    diagnosis_dict = {0: 'No_DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferate_DR'}
    df['binary_type'] = df['diagnosis'].map(diagnosis_dict_binary.get)
    df['type'] = df['diagnosis'].map(diagnosis_dict.get)
    return df

def plot_value_counts(df, column, title):
    """Plot horizontal bar chart of value counts."""
    fig, ax = plt.subplots()
    df[column].value_counts().plot(kind='barh', ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def load_image(image_path):
    """Load an image using cv2 and convert from BGR to RGB."""
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def display_sample_image(img_path, title):
    """Display a sample image with a given title."""
    img = load_image(img_path)
    if img is not None:
        st.image(img, caption=title, use_column_width=True)
    else:
        st.error(f"Could not load image: {img_path}")

def get_dataframe_from_image_folder(folder_path):
    """
    Create a DataFrame with filepaths and labels from the images folder.
    The folder structure should be: folder_path/label_name/image_file
    """
    filepaths = []
    labels = []
    classlist = os.listdir(folder_path)
    for klass in classlist:
        classpath = os.path.join(folder_path, klass)
        if os.path.isdir(classpath):
            flist = os.listdir(classpath)
            for f in flist:
                fpath = os.path.join(classpath, f)
                filepaths.append(fpath)
                labels.append(klass)
    df_local = pd.concat([pd.Series(filepaths, name='filepaths'),
                          pd.Series(labels, name='labels')], axis=1)
    return df_local

def get_data_generators(df, img_size, batch_size=40, test_batch_size=40,
                        train_split=0.8, valid_split=0.1):
    """
    Split the data into train, validation, and test sets.
    Create ImageDataGenerators for each.
    """
    dummy_split = valid_split / (1 - train_split)
    train_df, dummy_df = train_test_split(df, train_size=train_split, shuffle=True, random_state=123)
    valid_df, test_df = train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=123)
    def scalar(img):
        return img
    trgen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
    tvgen = ImageDataGenerator(preprocessing_function=scalar)
    train_gen = trgen.flow_from_dataframe(
        train_df, x_col='filepaths', y_col='labels', target_size=img_size,
        class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
    test_gen = tvgen.flow_from_dataframe(
        test_df, x_col='filepaths', y_col='labels', target_size=img_size,
        class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)
    valid_gen = tvgen.flow_from_dataframe(
        valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
        class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
    return train_gen, valid_gen, test_gen, train_df, valid_df, test_df

# ---------------------------
# Functions for Displaying Training Plots and Evaluation
# ---------------------------
def display_accuracy(history):
    """Plot training and validation accuracy."""
    fig, ax = plt.subplots()
    ax.plot(history.history['acc'], label='Train Accuracy')
    ax.plot(history.history['val_acc'], label='Validation Accuracy')
    ax.set_title('Model Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)

def display_loss(history):
    """Plot training and validation loss."""
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Model Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

def plot_confusion_matrix(y_true, y_pred, classes):
    """Display a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# ---------------------------
# (Optional) Callback Example: LRA
# ---------------------------
class LRA(keras.callbacks.Callback):
    reset = False
    count = 0
    stop_count = 0

    def __init__(self, model, patience, stop_patience, threshold, factor,
                 dwell, model_name, freeze, batches, initial_epoch, epochs, ask_epoch):
        super(LRA, self).__init__()
        self._model = model
        self.patience = patience
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.factor = factor
        self.dwell = dwell
        self.model_name = model_name
        self.freeze = freeze
        self.batches = batches
        self.initial_epoch = initial_epoch
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.lr = float(tf.keras.backend.get_value(model.optimizer.lr))
        self.highest_tracc = 0.0
        self.lowest_vloss = np.inf
        self.best_weights = self.model.get_weights()

    @property
    def model(self):
        return self._model

    def on_train_begin(self, logs=None):
        st.text("Training started...")

    def on_epoch_end(self, epoch, logs=None):
        st.text(f"Epoch {epoch+1} ended.")

# ---------------------------
# Model Building Functions
# ---------------------------
def build_efficientnet_model(img_shape, num_classes=5):
    """
    Build model using EfficientNetB1 with Imagenet weights.
    """
    base_model = tf.keras.applications.EfficientNetB1(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
        pooling='max'
    )
    x = base_model.output
    x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = layers.Dense(
        256,
        kernel_regularizer=regularizers.l2(0.016),
        activity_regularizer=regularizers.l1(0.006),
        bias_regularizer=regularizers.l1(0.006),
        activation='relu'
    )(x)
    x = layers.Dropout(rate=0.45, seed=123)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_simple_cnn_model(img_shape, num_classes=5):
    """
    Build a simple sequential CNN model.
    """
    model = keras.Sequential([
        keras.Input(shape=img_shape),
        layers.Conv2D(8, (3, 3), padding="valid", activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(16, (3, 3), padding="valid", activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(32, (4, 4), padding="valid", activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model

# ---------------------------
# Inference: print_info function (as in original code)
# ---------------------------
def print_info(test_gen, preds, print_code, save_dir, subject):
    class_dict = test_gen.class_indices
    labels = test_gen.labels
    file_names = test_gen.filenames 
    error_list = []
    true_class = []
    pred_class = []
    prob_list = []
    new_dict = {v: k for k, v in class_dict.items()}
    classes = list(new_dict.values())
    dict_as_text = str(new_dict)
    dict_name = subject + '-' + str(len(classes)) + '.txt'
    dict_path = os.path.join(save_dir, dict_name)
    with open(dict_path, 'w') as x_file:
        x_file.write(dict_as_text)
    errors = 0      
    y_pred = []
    error_indices = []
    for i, p in enumerate(preds):
        pred_index = np.argmax(p)
        true_index = labels[i]
        if pred_index != true_index:
            error_list.append(file_names[i])
            true_class.append(new_dict[true_index])
            pred_class.append(new_dict[pred_index])
            prob_list.append(p[pred_index])
            error_indices.append(true_index)
            errors += 1
        y_pred.append(pred_index)
    if print_code != 0:
        if errors > 0:
            r = errors if print_code > errors else print_code
            st.text('{0:^28s}{1:^28s}{2:^28s}{3:^16s}'.format('Filename', 'Predicted Class', 'True Class', 'Probability'))
            for i in range(r):
                split1 = os.path.split(error_list[i])
                split2 = os.path.split(split1[0])
                fname = split2[1] + '/' + split1[1]
                st.text('{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(fname, pred_class[i], true_class[i], ' ', prob_list[i]))
        else:
            st.success("With accuracy of 100% there are no errors to print")
    if errors > 0:
        plot_bar = []
        plot_class = []
        for key, value in new_dict.items():
            count = error_indices.count(key)
            if count != 0:
                plot_bar.append(count)
                plot_class.append(value)
        plt.figure(figsize=(10, len(plot_class)/3))
        for i in range(len(plot_class)):
            plt.barh(plot_class[i], plot_bar[i])
        plt.title('Errors by Class on Test Set')
        st.pyplot(plt)
    y_true = np.array(labels)
    y_pred = np.array(y_pred)
    if len(classes) <= 30:
        cm = confusion_matrix(y_true, y_pred)
        length = len(classes)
        fig_width = 8 if length < 8 else int(length * 0.5)
        fig_height = 8 if length < 8 else int(length * 0.5)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(length)+0.5, classes, rotation=90)
        plt.yticks(np.arange(length)+0.5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(plt)
    clr = classification_report(y_true, y_pred, target_names=classes)
    st.text("Classification Report:\n" + clr)

# ---------------------------
# Streamlit App Main Section
# ---------------------------
st.title("Retinopathy Detection Web App")

# Sidebar inputs for file paths and Google Drive links
st.sidebar.header("Paths and Google Drive Links")
csv_path = st.sidebar.text_input("Path to train.csv",
                                 value="D:/College/Jupyter Lab/Major Project/train.csv")
# For the gaussian filtered images folder, you can enter a local path
gaussian_images_folder = st.sidebar.text_input("Path to gaussian_filtered_images folder",
                                               value="D:/College/Jupyter Lab/Major Project/gaussian_filtered_images")

# Google Drive Links for files
efficientnet_link_shared = st.sidebar.text_input("Link to efficientnetb1.h5",
                                                 value="https://drive.google.com/file/d/1itXKJUJ8jcn6t0Hy7tAPevFy7BDWX7TR/view?usp=drive_link")
retinopathy_model_link_shared = st.sidebar.text_input("Link to retinopathy_model.h5",
                                                      value="https://drive.google.com/file/d/1WJfYnAY639vXLAkRBH75XiiBhw5mgIQu/view?usp=drive_link")

# Convert shared links to direct download URLs
efficientnet_direct_url = convert_gdrive_url(efficientnet_link_shared)
retinopathy_direct_url = convert_gdrive_url(retinopathy_model_link_shared)

# Optionally download the models if they are not present locally.
efficientnet_model_local = "efficientnetb1.h5"
retinopathy_model_local = "retinopathy_model.h5"

if st.sidebar.button("Download Model Files"):
    download_file(efficientnet_direct_url, efficientnet_model_local)
    download_file(retinopathy_direct_url, retinopathy_model_local)
    st.success("Models downloaded successfully!")

# Sidebar mode selection
app_mode = st.sidebar.selectbox("Select App Mode",
                                ["Dataset Overview", "Preview Images", "Train Model", "Inference"])

# ---------------------------
# Mode 1: Dataset Overview
# ---------------------------
if app_mode == "Dataset Overview":
    st.header("Dataset Overview")
    try:
        df_csv = load_csv(csv_path)
        st.write("### CSV Head")
        st.write(df_csv.head())
        st.write("### DR Type Distribution")
        plot_value_counts(df_csv, 'type', "Distribution of DR Types")
        st.write("### Binary DR Distribution")
        plot_value_counts(df_csv, 'binary_type', "Binary DR Distribution")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")

# ---------------------------
# Mode 2: Preview Images
# ---------------------------
elif app_mode == "Preview Images":
    st.header("Preview Sample Images")
    # List sub-folders inside the gaussian images folder:
    if os.path.exists(gaussian_images_folder):
        classes = [d for d in os.listdir(gaussian_images_folder) 
                   if os.path.isdir(os.path.join(gaussian_images_folder, d))]
        if classes:
            selected_class = st.selectbox("Select DR Class", classes)
            class_folder = os.path.join(gaussian_images_folder, selected_class)
            image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                sample_img = os.path.join(class_folder, image_files[0])
                display_sample_image(sample_img, f"Sample: {selected_class}")
            else:
                st.warning("No images found in the selected folder.")
        else:
            st.error("No class folders found; check your folder path.")
    else:
        st.error("The gaussian images folder path does not exist.")

# ---------------------------
# Mode 3: Train Model
# ---------------------------
elif app_mode == "Train Model":
    st.header("Train the Model")
    # Create dataframe from gaussian filtered images folder
    if os.path.exists(gaussian_images_folder):
        df_images = get_dataframe_from_image_folder(gaussian_images_folder)
        st.write("### Images Distribution")
        st.write(df_images['labels'].value_counts())
    else:
        st.error("The provided images folder path does not exist.")
        st.stop()

    # Sub-sample images to a maximum (if needed)
    max_size = 1500
    sample_list = []
    groups = df_images.groupby('labels')
    for label in df_images['labels'].unique():
        group = groups.get_group(label)
        sample_count = len(group)
        if sample_count > max_size:
            samples = group.sample(max_size, random_state=123).reset_index(drop=True)
        else:
            samples = group.sample(frac=1.0, random_state=123).reset_index(drop=True)
        sample_list.append(samples)
    df_images = pd.concat(sample_list, axis=0).reset_index(drop=True)
    st.write("### After Sub-sampling")
    st.write(df_images['labels'].value_counts())

    # Data Augmentation directory (create an "aug" folder in the working directory)
    working_dir = os.getcwd()
    aug_dir = os.path.join(working_dir, 'aug')
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in df_images['labels'].unique():
        os.mkdir(os.path.join(aug_dir, label))
    
    # Data augmentation using ImageDataGenerator
    target = 1500
    gen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=0.2,
                             height_shift_range=0.2, zoom_range=0.2)
    groups = df_images.groupby('labels')
    for label in df_images['labels'].unique():
        group = groups.get_group(label)
        sample_count = len(group)
        if sample_count < target:
            aug_img_count = 0
            delta = target - sample_count
            target_dir = os.path.join(aug_dir, label)
            aug_gen = gen.flow_from_dataframe(
                group, x_col='filepaths', y_col=None, target_size=(224,224),
                class_mode=None, batch_size=1, shuffle=False, save_to_dir=target_dir,
                save_prefix='aug-', save_format='jpg'
            )
            while aug_img_count < delta:
                images = next(aug_gen)
                aug_img_count += len(images)
    
    # Combine original and augmented images
    aug_fpaths = []
    aug_labels = []
    for klass in os.listdir(aug_dir):
        classpath = os.path.join(aug_dir, klass)
        for f in os.listdir(classpath):
            aug_fpaths.append(os.path.join(classpath, f))
            aug_labels.append(klass)
    aug_df = pd.concat([pd.Series(aug_fpaths, name='filepaths'),
                         pd.Series(aug_labels, name='labels')], axis=1)
    ndf = pd.concat([df_images, aug_df], axis=0).reset_index(drop=True)
    
    # Split data into train, validation, test sets
    train_split = 0.8
    valid_split = 0.1
    dummy_split = valid_split / (1 - train_split)
    train_df, dummy_df = train_test_split(ndf, train_size=train_split, shuffle=True, random_state=123)
    valid_df, test_df = train_test_split(dummy_df, train_size=dummy_split, shuffle=True, random_state=123)
    st.write(f"Train: {len(train_df)}, Validation: {len(valid_df)}, Test: {len(test_df)}")
    
    # Image generator parameters
    height, width, channels = 224, 224, 3
    batch_size = 40
    img_shape = (height, width, channels)
    img_size = (height, width)
    # For simplicity, use fixed test batch size
    test_batch_size = 40
    train_gen, valid_gen, test_gen, _, _, _ = get_data_generators(ndf, img_size, batch_size, test_batch_size)
    
    # Display sample images from training generator
    st.write("### Sample Training Images")
    sample_images, _ = next(train_gen)
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axs[i].imshow(sample_images[i] / 255.)
        axs[i].axis('off')
    st.pyplot(fig)
    
    # Build model - you can choose either the EfficientNet-based model or the simple CNN
    st.write("Building EfficientNetB1-based model...")
    img_shape = (224,224,3)
    model = build_efficientnet_model(img_shape, num_classes=5)
    st.text("Model Summary:")
    model.summary(print_fn=st.text)

    # Train the model (for demo purposes, using 5 epochs here)
    if st.button("Start Training (5 epochs demo)"):
        with st.spinner("Training..."):
            history = model.fit(train_gen, epochs=5, validation_data=valid_gen)
        st.success("Training completed!")
        st.write("### Training Accuracy and Loss")
        display_accuracy(history)
        display_loss(history)
        
        # Save the model
        model_save_path = os.path.join(working_dir, "efficientnetb1_trained.h5")
        model.save(model_save_path)
        st.write(f"Model saved at: {model_save_path}")
        
        # Evaluate on test set
        st.write("### Evaluating on Test Set")
        preds = model.predict(test_gen)
        y_pred = np.argmax(preds, axis=1)
        y_true = test_gen.classes
        classes = list(train_gen.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        st.json(report)
        plot_confusion_matrix(y_true, y_pred, classes)
    
# ---------------------------
# Mode 4: Inference
# ---------------------------
elif app_mode == "Inference":
    st.header("Inference")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_resized = cv2.resize(img, (224,224))
        img_array = np.expand_dims(img_resized, axis=0)
        
        # Load the retinopathy model (download if needed)
        try:
            if not os.path.exists(retinopathy_model_local):
                download_file(retinopathy_direct_url, retinopathy_model_local)
            model = keras.models.load_model(retinopathy_model_local)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
        
        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction, axis=1)[0]
        label_map = {0: 'No_DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferate_DR'}
        st.write(f"**Predicted Class:** {label_map.get(pred_class, 'Unknown')}")
