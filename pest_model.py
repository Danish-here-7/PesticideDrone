from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def build_model(img_size=(224, 224)):
    rgb_input = Input(shape=(img_size[0], img_size[1], 3), name="rgb")
    thermal_input = Input(shape=(img_size[0], img_size[1], 1), name="thermal")
    meta_input = Input(shape=(3,), name="meta")  # assuming 3 metadata features

    # RGB pipeline
    x1 = Conv2D(16, (3,3), activation='relu')(rgb_input)
    x1 = MaxPooling2D()(x1)
    x1 = Flatten()(x1)

    # Thermal pipeline
    x2 = Conv2D(8, (3,3), activation='relu')(thermal_input)
    x2 = MaxPooling2D()(x2)
    x2 = Flatten()(x2)

    # Metadata pipeline
    x3 = Dense(16, activation='relu')(meta_input)

    # Combine
    combined = Concatenate()([x1, x2, x3])
    dense1 = Dense(64, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=[rgb_input, thermal_input, meta_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
