from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, Concatenate, Input

'''
UpSampling2D is just a simple scaling up of the image by using nearest 
neighbour or bilinear upsampling, so nothing smart. Advantage is it's cheap.

Conv2DTranspose is a convolution operation whose kernel is learnt 
(just like normal conv2d operation) while training your model. 
Using Conv2DTranspose will also upsample its input but the key difference is the 
model should learn what is the best upsampling for the job.

'''
# def build_unet(input_shape: tuple, num_classes: int) -> Model:

class Unet(layers.Layer):
    def __init__(self, image_shape: tuple, n_classes: int):
        super(layers.Layer).__init__()
        self.image_shape = image_shape
        self.n_classes = n_classes

        # Encoder
        self.e_l1c1 = Conv2D(16, (3,3), activation='relu', padding='same')
        self.e_l1c2 = Conv2D(16, (3,3), activation='relu', padding='same')
        self.e_l1mp = MaxPooling2D((2,2))
        self.e_l2c1 = Conv2D(32, (3,3), activation='relu', padding='same')
        self.e_l2c2 = Conv2D(32, (3,3), activation='relu', padding='same')
        self.e_l2mp = MaxPooling2D((2,2))
        self.e_l3c1 = Conv2D(64, (3,3), activation='relu', padding='same')
        self.e_l3c2 = Conv2D(64, (3,3), activation='relu', padding='same')
        self.e_l3mp = MaxPooling2D((2,2))
        self.e_l4c1 = Conv2D(128, (3,3), activation='relu', padding='same')
        self.e_l4c2 = Conv2D(128, (3,3), activation='relu', padding='same')
        self.e_l4mp = MaxPooling2D((2,2))

        # Latent
        self.l_c1 = Conv2D(256, (3,3), padding='same')
        self.l_c2 = Conv2D(256, (3,3), padding='same')
        
        # Decoder
        self.d_l4ct = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same') # BatchNormalization, Gelu
        self.d_l4c1 = Conv2D(128, (3,3), activation='relu', padding='same')
        self.d_l4c2 = Conv2D(128, (3,3), activation='relu', padding='same')
        self.d_l3ct = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')
        self.d_l3c1 = Conv2D(64, (3,3), activation='relu', padding='same')
        self.d_l3c2 = Conv2D(64, (3,3), activation='relu', padding='same')
        self.d_l2ct = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')
        self.d_l2c1 = Conv2D(32, (3,3), activation='relu', padding='same')
        self.d_l2c2 = Conv2D(32, (3,3), activation='relu', padding='same')
        self.d_l1ct = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')
        self.d_l1c1 = Conv2D(16, (3,3), activation='relu', padding='same')
        self.d_l1c2 = Conv2D(16, (3,3), activation='relu', padding='same')
        self.mask = Conv2D(n_classes, (1,1), activation='sigmoid')

    
    def call(self, x):
        e_l1 = self.e_l1c1(x)
        e_l1 = self.e_l1c2(e_l1)
        e_l2 = self.e_l1mp(e_l1)
        e_l2 = self.e_l2c1(e_l2)
        e_l2 = self.e_l2c2(e_l2)
        e_l3 = self.e_l3mp(e_l2)
        e_l3 = self.e_l3c1(e_l3)
        e_l3 = self.e_l3c2(e_l3)
        e_l4 = self.e_l4mp(e_l3)
        e_l4 = self.e_l4c1(e_l4)
        e_l4 = self.e_l4c2(e_l4)

        l_latent = self.e_l4mp(e_l4)
        l_latent = self.l_c1(l_latent)
        l_latent = self.l_c2(l_latent)

        d_l4 = self.d_l4ct(l_latent)
        d_l4 = Concatenate()([d_l4, e_l4])
        d_l4 = self.d_l4c1(d_l4)
        d_l4 = self.d_l4c2(d_l4)
        d_l3 = self.d_l3ct(d_l4)
        d_l3 = Concatenate()([d_l3, e_l3])
        d_l3 = self.d_l3c1(d_l3)
        d_l3 = self.d_l3c2(d_l3)
        d_l2 = self.d_l2ct(d_l3)
        d_l2 = Concatenate()([d_l2, e_l2])
        d_l2 = self.d_l2c1(d_l2)
        d_l2 = self.d_l2c2(d_l2)
        d_l1 = self.d_l1ct(d_l2)
        d_l1 = Concatenate()([d_l1, e_l1])
        d_l1 = self.d_l1c1(d_l1)
        d_l1 = self.d_l1c2(d_l1)
        d_l1 = self.mask(d_l1)

        return d_l1


    def build(self):
        images = Input(shape=self.image_shape, name="Images")
        masks = self.call(images)
        model = Model(images, masks)
        
        return model
        

