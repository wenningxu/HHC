import argparse

def resolve_text_drop_probability(args):
    """Resolve the training-only override without changing historical defaults."""
    value = getattr(args, 'text_drop_prob', None)
    if value is None:
        return 0.1
    if args.conditioning_contract != 'paper_text_cfg' or not 0. <= value <= 1.:
        raise ValueError('--text-drop-prob requires paper_text_cfg and a finite value in [0,1]')
    return value

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=str, default='cuda:1')
    ## dataloader
    
    parser.add_argument('--dataname', type=str, default='hhi', help='dataset directory')
    parser.add_argument('--batch-size', default=96, type=int, help='paper MGT batch size')
    parser.add_argument('--num-workers', default=8, type=int, help='dataloader workers')
    parser.add_argument('--fps', default=[30], nargs="+", type=int, help='frames per second')
    parser.add_argument('--seq-len', type=int, default=64, help='training motion length')
    parser.add_argument('--train-split-file', default='dataset/interx/splits/train.txt')
    parser.add_argument('--val-split-file', default='dataset/interx/splits/val.txt')
    parser.add_argument('--train-motion-file', default='dataset/interx/motions/train.h5')
    parser.add_argument('--val-motion-file', default='dataset/interx/motions/val.h5')
    parser.add_argument('--all-split-file', default='dataset/interx/splits/all.txt')
    parser.add_argument('--all-motion-file', default='dataset/interx/motions/all.h5')
    parser.add_argument('--stats-dir', default='dataset/interx')
    parser.add_argument('--stats-prefix', default='Interx')
    parser.add_argument(
        '--train-asset-root', default=None,
        help=(
            'Optional optimized-augmentation overlay containing betas/, '
            'contact/, and texts_processed/; official assets remain fallback.'
        ),
    )
    parser.add_argument(
        '--augmentation-manifest', default=None,
        help='Required complete materialization manifest when train-asset-root is set.',
    )
    
    ## optimization
    parser.add_argument('--total-iter', default=300000, type=int, help='number of total iterations to run')
    parser.add_argument(
        '--diagnostic-stop-iter', default=0, type=int,
        help=(
            'Diagnostic-only clean stop after the validation ending at this '
            'step; zero disables it and the optimizer schedule still uses total-iter.'
        ),
    )
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[150000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=256, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=8096, help="per-part codebook size reported in the paper")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument(
        "--shape-condition-dim", type=int, default=8,
        help="VQ decoder shape-condition width (the original 8096/64 checkpoint also uses 8)",
    )
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    parser.add_argument('--motion_rep', type=str, default='raw', help='dataset directory')


    ## gpt arch
    parser.add_argument("--block-size", type=int, default=51, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=512, help="paper transformer latent dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=9, help="nb of transformer layers")
    parser.add_argument("--num-local-layer", type=int, default=9, help="legacy option; paper blocks all use word cross-attention")
    parser.add_argument("--n-head-gpt", type=int, default=8, help="paper attention head count")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")
    parser.add_argument("--contact", action=argparse.BooleanOptionalAction, default=True,
                        help="apply paper contact logit guidance during sampling")
    parser.add_argument("--contact-conditioning", action=argparse.BooleanOptionalAction, default=False,
                        help="ablation only: feed contact as a model input (paper keeps this disabled)")
    parser.add_argument("--shape", action=argparse.BooleanOptionalAction, default=True,
                        help="condition on the two SMPL-X body shapes and genders")
    parser.add_argument("--paper-architecture", action=argparse.BooleanOptionalAction, default=True,
                        help="use nine cross-attention+self-attention AdaLN blocks")
    parser.add_argument(
        "--training-mask-contract",
        choices=[
            "paper_kinematic",
            "historical_flat_random_replace",
            "historical_pipeline_paper_kinematic_ratio",
        ],
        default="paper_kinematic",
        help=(
            "MGT diagnostic bridge. The default is the paper's independently "
            "sorted Root/Body/Hand cosine mask; the historical choice exactly "
            "reproduces the pre-cleanup flat mask plus random-token corruption; "
            "historical_pipeline_paper_kinematic_ratio preserves that historical "
            "corruption and changes only the three mask ratios."
        ),
    )
    parser.add_argument(
        "--training-loss-contract",
        choices=[
            "paper_masked_only",
            "paper_masked_global_token_mean",
            "historical_all_valid",
        ],
        default="paper_masked_only",
        help=(
            "MGT diagnostic bridge. Historical MMM averaged NLL over every "
            "valid token. Both paper variants score masked tokens only; the "
            "global-token variant matches released InterMask batch reduction."
        ),
    )
    parser.add_argument(
        "--optimizer-contract",
        choices=["paper_warmup_cosine", "historical_multistep"],
        default="paper_warmup_cosine",
        help=(
            "MGT diagnostic bridge. historical_multistep uses AdamW "
            "betas=(0.5,0.9) and the recorded MultiStepLR schedule."
        ),
    )
    parser.add_argument(
        "--conditioning-contract",
        choices=["paper_text_cfg", "historical_mmm_shape_dropout"],
        default="paper_text_cfg",
        help=(
            "Training-time conditioning dropout contract. Historical MMM "
            "retains word tokens while dropping global CLIP and shape; the "
            "paper contract drops text while retaining requested body shape."
        ),
    )
    parser.add_argument(
        "--training-token-length-contract",
        choices=["original_mmm_38", "paper_50", "historical_38"],
        default="original_mmm_38",
        help=(
            "Inter-X training-window provenance. original_mmm_38 restores "
            "the released MMM loader default: 38 tensor positions reserve "
            "one END slot and therefore retain 37 motion tokens/148 frames, "
            "versus the fair-evaluation cap of 150 frames; paper_50 retains "
            "the completed "
            "50-window baseline; historical_38 is the legacy replay alias."
        ),
    )
    parser.add_argument('--text-drop-prob', type=float, default=None,
                        help='Training-only joint text dropout override for paper_text_cfg; default preserves 0.1.')
    parser.add_argument(
        "--training-validation-contract",
        choices=["paper_isolated_common", "historical_test_side_effect"],
        default="paper_isolated_common",
        help=(
            "MGT diagnostic bridge. historical_test_side_effect reproduces the "
            "original test-split sanity/milestone validation and its intentional "
            "RNG side effects; it is never eligible for final reporting."
        ),
    )

    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume vq pth')
    parser.add_argument("--resume-trans", type=str, default=None, help='resume gpt pth')
    parser.add_argument("--resume-training-ckpt", type=str, default=None,
                        help='Lightning checkpoint for full optimizer/scheduler resume')
    parser.add_argument(
        '--resume-in-place', action='store_true',
        help=(
            'Resume a full Lightning checkpoint inside its existing run '
            'directory. The prior manifest, experiment identity and launch '
            'source hashes must still match exactly.'
        ),
    )
    parser.add_argument("--vq-checkpoint", type=str, default='checkpoints/vq_paper/net_150000.pth',
                        help='shape-aware VQ checkpoint')
    parser.add_argument("--codebook-dir", type=str, default='checkpoints/vq_paper/codebook',
                        help='cached tokenizer outputs used for MGT training')
    parser.add_argument(
        '--cache-root', default=None,
        help=(
            'Isolated root for contact_token_label and code_contact_probs.npy; '
            'defaults to the VQ checkpoint directory for legacy compatibility.'
        ),
    )
    parser.add_argument(
        '--token-cache-contract',
        choices=[
            'deterministic_raw_v1',
            'deterministic_historical_root_v1',
            'historical_mmm_exact',
            'hybrid_historical_original_plus_deterministic_augmented_v1',
        ],
        default='deterministic_raw_v1',
        help=(
            'Tokenization provenance. historical_mmm_exact imports the '
            'byte-exact pre-cleanup MMM token/contact cache while still '
            'restricting the training loader and contact map to train/val splits. '
            'The hybrid contract keeps those exact bytes for original train/val '
            'samples and tokenizes only materialized shape-augmentation samples.'
        ),
    )
    parser.add_argument(
        '--historical-codebook-source', default=None,
        help='Required source directory for historical_mmm_exact token imports.',
    )
    parser.add_argument(
        '--historical-contact-source', default=None,
        help='Required source directory for historical_mmm_exact contact-token imports.',
    )
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    parser.add_argument('--vq-name', type=str, default='VQVAE', help='name of the generated dataset .npy, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=10000, type=int, help='evaluation frequency')
    parser.add_argument('--checkpoint-iter', default=5000, type=int,
                        help='resumable latest-checkpoint frequency')
    parser.add_argument(
        '--save-all-validation-checkpoints',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Retain every validation-interval checkpoint. This is required '
            'when an historical test-side-effect hook is preserved only for '
            'trajectory fidelity and held-out validation will select the '
            'reported checkpoint post hoc.'
        ),
    )
    parser.add_argument('--seed', default=114514, type=int, help='seed for initializing training. ')
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=.5, help='keep rate for gpt training')
    
    ## generator
    parser.add_argument('--text', type=str, help='text')
    parser.add_argument('--length', type=int, help='length')
    parser.add_argument("--generate-only", action='store_true', help='only run generation/export')
    parser.add_argument("--evaluate-only", action='store_true', help='run repeated official Inter-X evaluation')
    parser.add_argument(
        "--cache-only", action='store_true',
        help='build and validate the VQ token/contact cache, then exit before MGT construction',
    )
    parser.add_argument("--gen-ckpt", type=str, default=None, help='checkpoint for t2i generation/export')
    parser.add_argument("--gen-split", type=str, default='val', choices=['val', 'test'], help='dataset split for generation/export')
    parser.add_argument("--num-repeat", type=int, default=30, help='number of multimodal samples per text')
    parser.add_argument("--save-dir", type=str, default=None, help='output directory for generation/export')
    parser.add_argument("--save-embeddings", action='store_true', help='save motion embeddings alongside motions')
    parser.add_argument("--eval-split", default='test', choices=['val', 'test'], help='split for official evaluation')
    parser.add_argument("--replication-times", default=20, type=int, help='official evaluation repetitions')
    parser.add_argument("--mm-num-samples", default=100, type=int,
                        help="official number of text prompts used for multimodality")
    parser.add_argument("--mm-num-repeats", default=30, type=int,
                        help="official generations per multimodality prompt")
    parser.add_argument(
        "--validation-mm-repeats", default=3, type=int,
        help=(
            "cheap periodic validation MModality diagnostic; final reporting "
            "continues to use --mm-num-repeats (normally 30)"
        ),
    )
    parser.add_argument("--eval-seed", default=1234, type=int,
                        help="base seed shared by all methods in fair evaluation")
    parser.add_argument(
        "--sampler", default='sample_intra_cas',
        choices=['sample', 'sample_intra_cas', 'sample_intra_cas_repredict'],
        help=(
            'Formal runs use sample_intra_cas. sample_intra_cas_repredict is '
            'a validation-only update-support diagnostic with the same outer '
            'sampling settings.'
        ),
    )
    parser.add_argument("--temperature", default=0.5, type=float, help='paper sampling temperature')
    parser.add_argument("--sampling-steps", default=50, type=int,
                        help='MaskGIT iterations for the requested common fair protocol; use 10 for the paper-native audit')
    parser.add_argument("--cfg-scale", default=3.5, type=float, help='classifier-free text guidance')
    parser.add_argument("--contact-guidance-scale", default=1.0, type=float, help='paper contact logit-bias weight')
    parser.add_argument("--wandb-mode", default='online', choices=['online', 'offline', 'disabled'])

    args = parser.parse_args()
    try:
        resolve_text_drop_probability(args)
    except ValueError as error:
        parser.error(str(error))
    return args
