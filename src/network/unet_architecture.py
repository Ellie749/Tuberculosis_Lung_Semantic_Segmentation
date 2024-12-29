from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D

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
        self.e_l4MP = MaxPooling2D((2,2))

        # Latent
        self.l_c1 = Conv2D(256, (3,3), padding='same')
        self.l_c2 = Conv2D(256, (3,3), padding='same')
        
        # Decoder
        self.d_l4ct = Conv2DTranspose(128, (3,3), padding='same') #stride?
        self.d_l4c1 = Conv2D(128, (3,3), activation='relu', padding='same')
        self.d_l4c2 = Conv2D(128, (3,3), activation='relu', padding='same')
        self.d_l3ct = Conv2DTranspose(64, (3,3), padding='same')
        self.d_l3c1 = Conv2D(64, (3,3), activation='relu', padding='same')
        self.d_l3c2 = Conv2D(64, (3,3), activation='relu', padding='same')
        self.d_l2ct = Conv2DTranspose(64, (3,3), padding='same')
        self.d_l2c1 = Conv2D(32, (3,3), activation='relu', padding='same')
        self.d_l2c2 = Conv2D(32, (3,3), activation='relu', padding='same')
        self.d_l1ct = Conv2DTranspose(64, (3,3), padding='same')
        self.d_l1c1 = Conv2D(16, (3,3), activation='relu', padding='same')
        self.d_l1c2 = Conv2D(16, (3,3), activation='relu', padding='same')
        self.mask = Conv2D(n_classes, (1,1), padding='same')

    
    def call(self, x):
        l = self.e_l1c1(x)


    def build(self):
        pass
        

