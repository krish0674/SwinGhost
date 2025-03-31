import argparse
from utils.trainer import train_model
import wandb


def main(args):
    config = {
        'batch_size': args.batch_size,
        'root_dir' : args.root_dir,
        'epochs': args.epochs,
        'dset' : args.dset,
        'device':args.device,
        'lr': args.lr,
        'loss_weight':args.loss_weight,
        'gan_type':args.gan_type,
        'sf_path' : args.sf_path,
    }
    wandb.login(key=args.key)
    wandb.init(project="LapLoss",
               config={'lr':args.lr, 'max_ssim':0, 'max_psnr':0,'test_psnr':0,'test_ssim': 0, 'best_epoch':0, 'loss_weight':args.loss_weight, 'gan_type': args.gan_type}, allow_val_change=True)
    train_model(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--epochs', type=int, required=False, default=300)
    parser.add_argument('--root_dir', type=str, required=False)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    parser.add_argument('--dset', type=str, required=False, default='sice')
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--loss_weight', type=float, required=False, default=3000)
    parser.add_argument('--gan_type', type=str, required=False,default='vanilla')
    parser.add_argument('--key', type=str, required=False, default='6a0d17eac1d2d0d31cb4921178660ec68b3b40e5')
    parser.add_argument('--sf_path', type=str, required=False, default='./best_model_g.pth')
    arguments = parser.parse_args()
    main(arguments)
