#Importing dependencies
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD

from models.resnet34 import ResNet34
from data.cifar10 import load_cifar10



def main():
    IMG_SIZE = 224         
    BATCH_SIZE = 32
    EPOCHS = 30             
    NUM_CLASSES = 10
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4

    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    #Loading CIFAR10
    train_ds, test_ds = load_cifar10(
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    #Building Model
    model = ResNet34(num_classes=NUM_CLASSES)

    model.build((None, IMG_SIZE, IMG_SIZE, 3))
    model.summary()


    optimizer = SGD(
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    #Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "resnet34_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=3,
            min_lr=1e-5
        )
    ]

    #Training
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
