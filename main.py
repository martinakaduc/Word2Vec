from keras.layers import Input, Lambda, Activation
from keras.models import Model
from keras.layers.embeddings import Embedding
import keras.backend as K
import utils
import argparse
import random

def batch_generator(cpl, lbl, batch_size, nb_batch):
    # trim the tail
    garbage = len(lbl) % batch_size

    pvt = cpl[:, 0][:-garbage]
    ctx = cpl[:, 1][:-garbage]
    lbl = lbl[:-garbage]

    assert pvt.shape == ctx.shape == lbl.shape

    # epoch loop
    while 1:
        # shuffle data at beginning of every epoch (takes few minutes)
        seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(pvt)
        random.seed(seed)
        random.shuffle(ctx)
        random.seed(seed)
        random.shuffle(lbl)

        for i in range(nb_batch):
            begin, end = batch_size*i, batch_size*(i+1)
            # feed i th batch
            yield ([pvt[begin: end], ctx[begin: end]], lbl[begin: end])

def build_model(vocab_size, vec_dim, batch_size):
    # graph definition (pvt: center of window, ctx: context)
    input_pvt = Input(batch_shape=(batch_size, 1), dtype='int32')
    input_ctx = Input(batch_shape=(batch_size, 1), dtype='int32')

    embedded_pvt = Embedding(input_dim=vocab_size,
                             output_dim=vec_dim,
                             input_length=1)(input_pvt)

    embedded_ctx = Embedding(input_dim=vocab_size,
                             output_dim=vec_dim,
                             input_length=1)(input_ctx)

    merged = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1))([embedded_pvt, embedded_ctx])

    predictions = Activation('sigmoid')(merged)


    # build and train the model
    model = Model(inputs=[input_pvt, input_ctx], outputs=predictions)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    model.summary()

    return model

def main(args):
    sentences, index2word = utils.load_corpus()
    vocab_size = len(index2word)

    # create input
    couples, labels = utils.skip_grams(sentences, args.window_size, vocab_size)
    print('Shape of couples: ' + str(couples.shape))
    print('Shape of labels: ' + str(labels.shape))

    # metrics
    nb_batch = len(labels) // args.batch_size
    samples_per_epoch = args.batch_size * nb_batch

    model = build_model(vocab_size, args.vec_dim, args.batch_size)

    model.fit_generator(generator=batch_generator(couples, labels, args.batch_size, nb_batch),
                        steps_per_epoch=samples_per_epoch,
                        epochs=args.epochs, verbose=1)

    # save weights
    utils.save_weights(model, index2word, args.vec_dim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='truyen_kieu.txt')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--vec_dim', type=int, default=300)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=61440)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--ckpt_period', type=int, default=1)

    args = parser.parse_args()
    main(args)
