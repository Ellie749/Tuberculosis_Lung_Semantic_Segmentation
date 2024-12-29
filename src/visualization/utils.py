import matplotlib.pyplot as plt


def plot_metrics(H):
    
    plt.subplot(121)
    plt.title("LOSS")
    plt.plot(range(len(H.history['loss'])), H.history['loss'], label='train loss')
    plt.plot(range(len(H.history['val_loss'])), H.history['val_loss'], label='validation loss')

    plt.subplot(122)
    plt.title("ACCURACY")
    plt.plot(range(len(H.history['accuracy'])), H.history['accuracy'], label='train accuracy')
    plt.plot(range(len(H.history['val_accuracy'])), H.history['val_accuracy'], label='validation accuracy')

    plt.legend()
    plt.savefig("metrics.png")
    plt.show()