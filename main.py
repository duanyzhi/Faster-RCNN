from lib.network.net import net


def run():
    model = net('train')
    model.resnet_rpn()
    sess = model.gpu_config()

    model.train(sess)

run()

