from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input, BatchNormalization, Activation

class KonversiAksaraModel():
    def buat_model(self):
        input_shape = (50, 50, 1)
        inputs = Input(shape=input_shape)
        induk_branch = self.build_induk_branch(inputs, 20)
        anak_branch = self.build_anak_branch(inputs, 13)
        model = Model(inputs=inputs,
                      outputs = [induk_branch, anak_branch],
                      name="konversi_aksara")
        init_lr = 1e-4
        epochs = 10
        opt = Adam(init_lr, decay=init_lr / epochs)
        model.compile(optimizer=opt, 
                      loss = {
                          'induk_output': 'sparse_categorical_crossentropy', 
                          'anak_output': 'sparse_categorical_crossentropy'
                          },
                      metrics = {
                          'induk_output': 'accuracy',
                          'anak_output': 'accuracy'
                          })
        return model
    def make_default_hidden_layers(self, inputs):
        x = Conv2D(32, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.35)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x)
        return x
    def build_induk_branch(self, inputs, jumlah_induk):
        x = self.make_default_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(750)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.45)(x)
        x = Dense(500)(x)
        x = Activation("relu")(x)
        x = Dense(jumlah_induk)(x)
        x = Activation("softmax", name="induk_output")(x)
        return x
    def build_anak_branch(self, inputs, jumlah_anak):
        x = self.make_default_hidden_layers(inputs)
        x = Flatten()(x)
        x = Dense(750)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.45)(x)
        x = Dense(500)(x)
        x = Activation("relu")(x)
        x = Dense(jumlah_anak)(x)
        x = Activation("softmax", name="anak_output")(x)
        return x