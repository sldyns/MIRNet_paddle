import paddle
import paddle.optimizer as optim
import paddle.distributed as dist
from networks.MIRNet_model import MIRNet

def main():
    dist.init_parallel_env()
    nranks = paddle.distributed.ParallelEnv().nranks

    print(nranks)
    model = MIRNet()

    model.train()

    new_lr = 2e-4
    scheduler = optim.lr.CosineAnnealingDecay(learning_rate=new_lr, T_max=60, eta_min=1e-6)
    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=scheduler, weight_decay=1e-8)

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object

    for epoch in range(270):
        # update lr
        if isinstance(optimizer, paddle.distributed.fleet.Fleet):
            lr_sche = optimizer.user_defined_optimizer._learning_rate
        else:
            lr_sche = optimizer._learning_rate
        if isinstance(lr_sche, paddle.optimizer.lr.LRScheduler):
            lr_sche.step()

        if (epoch + 1) == 60:
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                optimizer.user_defined_optimizer._learning_rate = optim.lr.CosineAnnealingDecay(learning_rate=1e-4,
                                                                                                T_max=60,
                                                                                                eta_min=1e-6)
            else:
                optimizer._learning_rate = optim.lr.CosineAnnealingDecay(learning_rate=1e-4, T_max=60, eta_min=1e-6)

        if (epoch + 1) == 120:
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                optimizer.user_defined_optimizer._learning_rate = optim.lr.CosineAnnealingDecay(learning_rate=1e-4,
                                                                                                T_max=60,
                                                                                                eta_min=1e-6)
            else:
                optimizer._learning_rate = optim.lr.CosineAnnealingDecay(learning_rate=1e-4, T_max=60, eta_min=1e-6)

        if (epoch + 1) == 180:
            if isinstance(optimizer, paddle.distributed.fleet.Fleet):
                optimizer.user_defined_optimizer._learning_rate = optim.lr.CosineAnnealingDecay(learning_rate=8e-5,
                                                                                                T_max=90,
                                                                                                eta_min=1e-6)
            else:
                optimizer._learning_rate = optim.lr.CosineAnnealingDecay(learning_rate=8e-5, T_max=90, eta_min=1e-6)

        print("Epoch {}\tLearningRate {:.6f}".format(epoch, optimizer.get_lr()))


if __name__ == '__main__':
    main()

