def set_arguments(args):

    args.net_type = 'convnet'
    args.depth = 3
    args.niter = 10000
    if args.dataset[:5] == 'cifar':
        args.metric = 'mse'
        args.lr_img = 0.1
    elif args.dataset == 'svhn':
        args.metric = 'mse'
        args.lr_img = 0.1
        if args.factor == 1 and args.ipc == 1:
            args.mixup = 'vanilla'
            args.dsa_strategy = 'color_crop_cutout_scale_rotate'
    elif args.dataset == 'mnist':
        args.metric = 'mse'
        args.lr_img = 0.1
        if args.factor > 1:
            args.aug_type = 'color_crop'
            args.mixup_net = 'vanilla'
            args.mixup = 'vanilla'
            args.dsa_strategy = 'color_crop_scale_rotate'
    elif args.dataset == 'fashion':
        args.metric = 'mse'
        args.lr_img = 0.1
    else:
        raise AssertionError("Not supported dataset!")


    log = f"Arguments are loaded!"
    log += f", net: {args.net_type}-{args.depth}"
    log += f", metric: {args.metric}"
    log += f", lr_img: {args.lr_img}"
    if args.decode_type != 'single':
        log += f", decode: {args.decode_type}"
    print(log)

    return args
