import math
from tensorboardX import SummaryWriter
from dataset import get_dataloader
from conf import settings
from utils import *
import function
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def main(args, rank=0):
    # create logger & writer
    args.path_helper = None
    if rank == 0:
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        logger.info(args)
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)
        writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))

    # create network
    net = get_network(args, args.net, args.proj_type)
    # resume if args.weights
    net = resume_weights(net, args.weights)

    # set trainable parameters
    trainable_params_names = set_trainable_params(net, net_name=args.net)

    # put network to multi-gpu or single gpu
    # must use torch.cuda.set_device(rank) https://github.com/pytorch/pytorch/issues/21819#issuecomment-553310128
    torch.cuda.set_device(rank)
    net = net.cuda()
    if args.ddp:
        net = DDP(net, device_ids=[rank], find_unused_parameters=True)

    # optim & scheduler
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # learning rate decay
    # resume if args.weights
    optimizer, scheduler, start_epoch = resume_optim_scheduler_epoch(optimizer, scheduler, args.weights)

    # dataloaders
    train_dataloader, test_dataloader, train_sampler, test_sampler = get_dataloader(args)

    # begin training
    best_iou = 0
    for epoch in range(start_epoch, settings.EPOCH):
        if args.ddp:
            dist.barrier()
            train_sampler.set_epoch(epoch)
        # eval
        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH - 1:
            tol, eiou, edice = function.validation_sam(args, test_dataloader, epoch, net, vis=args.vis)

            # print & save
            if rank == 0:
                logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
                if eiou > best_iou:
                    best_iou = eiou
                    full_state_dict = net.module.state_dict() if args.ddp else net.state_dict()
                    saved_state_dict = {k: v for k, v in full_state_dict.items() if k in trainable_params_names}
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model': args.net,
                        'state_dict': saved_state_dict,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_iou': best_iou,
                        'path_helper': args.path_helper,
                    }, is_best=True, output_dir=args.path_helper['ckpt_path'], filename=f"last_checkpoint.pth")

        # train
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, train_dataloader, epoch, vis=args.vis)

        scheduler.step()
        time_end = time.time()
        if rank == 0:
            logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
            print('time_for_training ', time_end - time_start)

    # finish training
    if rank == 0:
        writer.close()


def cleanup():
    dist.destroy_process_group()


def ddp_main(rank, args, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    main(args, rank=rank)
    cleanup()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    args = cfg.parse_args()
    set_seed(args.seed)
    if args.ddp:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(12345+random.randint(0,1000))
        world_size = torch.cuda.device_count()
        # !!!! In ddp, the batch size is node-level
        args.b = math.ceil(args.b / world_size)
        mp.spawn(ddp_main,
                 args=(args, world_size),
                 nprocs=world_size,
                 join=True)
    else:
        main(args)
