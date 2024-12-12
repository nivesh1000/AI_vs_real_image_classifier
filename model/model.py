from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
import tensorflow as tf
import os

def create_data_generator(directory: str, target_size: tuple, batch_size: int, 
                          class_mode: str, shuffle: bool) -> tf.keras.preprocessing.image.DirectoryIterator:
    """
    Create a data generator for the given directory.
    Args:
        directory: Path to the dataset directory.
        target_size: Tuple indicating the target size of the images.
        batch_size: Number of images per batch.
        class_mode: Mode of classification (e.g., 'binary', 'categorical').
        shuffle: Whether to shuffle the data.
    Returns:
        A DirectoryIterator object for the dataset.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle
    )

def prepare_data_generators():
    """
    Create data generators for training, validation, and testing datasets.
    Returns:
        train_generator: Generator for training data.
        val_generator: Generator for validation data.
        test_generator: Generator for testing data.
    """
    train_val_generator = lambda dir_path: create_data_generator(
        directory=dir_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=True
    )

    train_generator = train_val_generator('Dataset/train')
    val_generator = train_val_generator('Dataset/valid')

    test_generator = create_data_generator(
        directory='Dataset/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def build_model() -> Sequential:
    """
    Build a DenseNet121-based sequential model for binary classification.
    Returns:
        model: Compiled Keras model.
    """
    base_model = DenseNet121(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )

    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.6),
        Dense(32, activation='relu'),
        Dropout(0.6),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model: Sequential, train_gen, val_gen) -> tf.keras.callbacks.History:
    """
    Train the model using training and validation generators.
    Args:
        model: Compiled Keras model.
        train_gen: Training data generator.
        val_gen: Validation data generator.
    Returns:
        history: Training history.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        callbacks=[early_stopping]
    )

    return history

def save_model(model: Sequential, save_path: str):
    """
    Save the trained model to the specified path.
    Args:
        model: Trained Keras model.
        save_path: Path to save the model.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved at {save_path}")

def evaluate_model(model: Sequential, test_gen):
    """
    Evaluate the model using the test data generator.
    Args:
        model: Trained Keras model.
        test_gen: Test data generator.
    """
    predictions = model.predict(test_gen)
    predicted_classes = (predictions > 0.5).astype(int)

    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes)
    recall = recall_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes)

    print("Test Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes,
                                target_names=class_labels))

if __name__ == "__main__":
    train_gen, val_gen, test_gen = prepare_data_generators()
    model = build_model()
    model.summary()
    history = train_model(model, train_gen, val_gen)
    save_model(model, "saved_models/densenet_model.h5")
    evaluate_model(model, test_gen)
