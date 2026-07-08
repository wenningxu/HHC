import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default='cuda:0')
    ## dataloader
    parser.add_argument('--dataname', type=str, default='hhi', help='dataset directory')
    parser.add_argument('--batch-size', default=512, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')
    parser.add_argument('--num_primitive', type=int, default=4, help='training motion length')
    parser.add_argument('--history_length', type=int, default=3, help='training motion length')
    parser.add_argument('--future_length', type=int, default=12, help='training motion length')

    ## diffusion
    parser.add_argument('--diffusion-steps', type=int, default=1000, help='number of diffusion steps')
    parser.add_argument('--respacing', type=str, default='ddim50', help='beta start for diffusion process')
    parser.add_argument("--noise_schedule", default="cosine")
    parser.add_argument("--sigma_small", default=True)
    parser.add_argument('--clip-version', type=str, default='ViT-B/32', help='conditioning mode: gt / predicted')

    ## vae
    parser.add_argument("--code-dim", type=int, default=256, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=3*4096, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')

    ## training
    parser.add_argument('--total-iter', default=300000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--stage1_steps', default=100000, type=float, help="learning rate decay")
    parser.add_argument('--stage2_steps', default=200000, type=float, help="learning rate decay")
    parser.add_argument('--recons-loss', type=str, default='l1_smooth', help='reconstruction loss')

    parser.add_argument('--weight-decay', default=0.00002, type=float, help='weight decay')
    parser.add_argument("--weight_latent_rec", type=float, default=1, help="hyper-parameter for the weight_latent_rec loss")
    parser.add_argument('--weight_feature_rec', type=float, default=1, help='hyper-parameter for the weight_feature_rec loss')
    parser.add_argument('--weight_velocity', type=float, default=100, help='hyper-parameter for the weight_feature_rec loss')
    parser.add_argument('--weight_rotation', type=float, default=100, help='hyper-parameter for the weight_feature_rec loss')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='ema')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='grad_clip')
    parser.add_argument('--rescale_latent', type=bool, default=True, help='rescale_latent')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')

    ## output directory
    parser.add_argument('--out-dir', type=str, default='output', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug',
                        help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=5000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')

    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')

    parser.add_argument('--sep-uplow', action='store_true', help='whether visualize GT motions')

    return parser.parse_args()