from lib.network.net import net


def run():
    model = net('train')
    train_step, loss = model.build_faster_rcnn()
    sess = model.gpu_config()

    model.train(sess, train_step, loss)

run()

