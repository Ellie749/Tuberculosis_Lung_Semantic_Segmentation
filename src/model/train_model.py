


#add IOU and other metrics
def train(model, train_images, train_masks, validation_images, validation_masks, epochs, batch_size):

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    H = model.fit(train_images, train_masks, validation_data=(validation_images, validation_masks), epochs=epochs, batch_size=batch_size)

    return H