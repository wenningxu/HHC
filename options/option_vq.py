import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Two-person VQ-VAE training',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default='cuda:0')
    ## dataloader
    parser.add_argument('--dataname', type=str, default='hhi', help='dataset name (hhi=InterX two-person)')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')
    parser.add_argument('--num_primitive', type=int, default=4, help='training motion length')
    parser.add_argument('--history_length', type=int, default=3, help='training motion length')
    parser.add_argument('--future_length', type=int, default=12, help='training motion length')
    parser.add_argument('--motion_rep', type=str, default='raw', help='motion root representation')
    parser.add_argument('--train-split-file', default='dataset/interx/train.txt')
    parser.add_argument('--train-motion-file', default='dataset/interx/motions/train.h5')
    parser.add_argument('--validation-split-file', default='dataset/interx/splits/val.txt')
    parser.add_argument('--validation-motion-file', default='dataset/interx/motions/val.h5')
    parser.add_argument('--stats-dir', default='dataset/interx')
    parser.add_argument('--stats-prefix', default='Interx')

    ## optimization
    parser.add_argument('--total-iter', default=150000, type=int, help='paper VQ-VAE training steps')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument(
        '--schedule-mode',
        default='integrated_warmup_cosine',
        choices=('integrated_warmup_cosine', 'historical_separate_warmup_cosine'),
        help=(
            'Whether warm-up is included in total-iter (paper-text '
            'interpretation) or reproduced as the historical MMM run\'s '
            'separate 1..warm_up_iter-1 update phase before total-iter cosine '
            'updates'
        ),
    )
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[200000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--stage1_steps', default=100000, type=float, help="learning rate decay")
    parser.add_argument('--stage2_steps', default=200000, type=float, help="learning rate decay")


    parser.add_argument('--weight-decay', default=1e-6, type=float, help='paper AdamW weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--kl', type=float, default=1e-3, help='hyper-parameter for the kl loss')
    parser.add_argument('--recons-loss', type=str, default='l1_smooth', help='reconstruction loss')
    parser.add_argument('--loss_explicit', type=float, default=3, help='hyper-parameter for the explicit loss')#0.1
    parser.add_argument('--loss_vel', type=float, default=100, help='hyper-parameter for the velocity loss')
    parser.add_argument('--loss_root', type=float, default=0.1, help='hyper-parameter for the bone length loss')
    parser.add_argument('--loss_geo', type=float, default=0.01, help='hyper-parameter for the geodesic loss') #0.001
    parser.add_argument('--loss_fc', type=float, default=10, help='hyper-parameter for the foot contact loss')  # 100
    parser.add_argument('--loss_inter_oriental', type=float, default=0, help='hyper-parameter for the foot contact loss')  # 100
    parser.add_argument('--loss_distance_map', type=float, default=0.1, help='hyper-parameter for the foot contact loss')  # 100
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=256, help="96/96/64 body/hand/root latent dimensions")
    parser.add_argument("--nb-code", type=int, default=8096, help="per-part codebook size reported in the paper")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=256, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    parser.add_argument('--levels', default=[8,5,5,5,], help='fsq levels')

    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset', 'fsq'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')


    ## output directory
    parser.add_argument('--out-dir', type=str, default='output', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--checkpoint-dir', default='checkpoints/vq_paper', help='paper VQ checkpoints and token cache')
    parser.add_argument('--wandb-mode', default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--precision', default='fp32', choices=['fp32', 'bf16'])
    parser.add_argument(
        '--shape-mask-prob', default=0.1, type=float,
        help='Probability of masking the VQ decoder shape condition during training',
    )
    parser.add_argument(
        '--validation-seed', default=20260130, type=int,
        help='Fixed seed used by validation crops/captions without perturbing training RNG',
    )
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=5000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')

    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')

    parser.add_argument('--sep-uplow', action='store_true', help='use upper/lower ablation path')

    return parser.parse_args()
