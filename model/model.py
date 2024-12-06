from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Normalize pixel values and create generators for train and validation datasets
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    'Dataset/train',                # Path to training data
    target_size=(224, 224),         # Resize images to 224x224
    batch_size=32,                  # Load images in batches of 32
    class_mode='binary'             # Binary classification
)

val_generator = val_datagen.flow_from_directory(
    'Dataset/valid',                # Path to validation data
    target_size=(224, 224),         # Resize images to 224x224
    batch_size=32,                  # Load images in batches of 32
    class_mode='binary'             # Binary classification
)

base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Unfreeze all layers in the base model (for fine-tuning)
for layer in base_model.layers:
    layer.trainable = True

model = Sequential([
    base_model,                             # Pre-trained ResNet50 base
    Flatten(),                              # Flatten the output from the base model
    Dense(320, activation='relu'),          # Fully connected layer with 320 nodes
    Dropout(0.5),                           # Dropout layer for regularization
    Dense(1, activation='sigmoid',          # Output layer for binary classification
          kernel_regularizer=l2(0.01))      # Add L2 regularization
])

model.compile(
    optimizer=Adam(learning_rate=0.001),    # Adam optimizer with a learning rate of 0.001
    loss='binary_crossentropy',            # Binary cross-entropy loss
    metrics=['accuracy']                   # Track accuracy during training
)

early_stopping = EarlyStopping(
    monitor='val_loss',                    # Monitor validation loss
    patience=3,                            # Stop after 3 epochs of no improvement
    restore_best_weights=True              # Restore weights from the best epoch
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,                             # Maximum number of epochs
    steps_per_epoch=len(train_generator),  # Total batches per epoch
    validation_steps=len(val_generator),   # Total batches for validation
    callbacks=[early_stopping]             # Use EarlyStopping during training
)
