import argparse
import os

def parse_args() :
    parser = argparse.ArgumentParser()
    ### Dataset
    parser.add_argument("--path_dataset", type=str, default="/data/krdu/dataset")

    ### Model S/L
    parser.add_argument("--load_path", type=str, default="/data/krdu/pretrained_models")
    parser.add_argument("--load_model", action='store_true', default=False)
    parser.add_argument("--load_name", type=str, default="")
    parser.add_argument("--save_path", type=str, default="/data/krdu/pretrained_models")
    parser.add_argument("--model_name", type=str, default="unnamed_model")
    parser.add_argument("--model_type", type=str, help="Which model to use (resnet18, vit_B, etc.)")

    ### Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--fine_tune_epochs", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default='AdamW')
    parser.add_argument("--scheduler", type=str, default='cosLR')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default="5")
    parser.add_argument("--workers", type=int, default="18")
    parser.add_argument("--data_parallel", action='store_true', default=False)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args

def parse_args_adni2_pet() :
    parser = argparse.ArgumentParser()
    ### Dataset
    parser.add_argument("--path_dataset", type=str, default="/data/krdu/dataset")

    ### Model S/L
    parser.add_argument("--load_path", type=str, default="/data/krdu/saved_models")
    parser.add_argument("--load_model", action='store_true', default=False)
    parser.add_argument("--load_name", type=str, default="")
    parser.add_argument("--save_path", type=str, default="/data/krdu/saved_models")
    parser.add_argument("--model_name", type=str, default="unnamed_model")
    parser.add_argument("--model_type", type=str, help="Which model to use (resnet18, vit_B, etc.)")

    ### Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_D", type=int, default=64)
    parser.add_argument("--input_H", type=int, default=64)
    parser.add_argument("--input_W", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--optimizer", type=str, default='AdamW')
    parser.add_argument("--scheduler", type=str, default='cosLR')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default="4")
    parser.add_argument("--workers", type=int, default="18")
    parser.add_argument("--data_parallel", action='store_true', default=False)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args


def parse_args_adni2_pet_mae() :
    parser = argparse.ArgumentParser()
    ### Dataset
    parser.add_argument("--path_dataset", type=str, default="/data/krdu/dataset")

    ### Model S/L
    parser.add_argument("--load_path", type=str, default="/data/krdu/pretrained_models")
    parser.add_argument("--load_model", action='store_true', default=False)
    parser.add_argument("--load_name", type=str, default="")
    parser.add_argument("--save_path", type=str, default="save_models/")
    parser.add_argument("--model_name", type=str, default="unnamed_model")
    parser.add_argument("--model_type", type=str, default='vit_B_mae')

    ### Training
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--input_D", type=int, default=128)
    parser.add_argument("--input_H", type=int, default=128)
    parser.add_argument("--input_W", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--optimizer", type=str, default='AdamW')
    parser.add_argument("--scheduler", type=str, default='cosLR')
    parser.add_argument("--lr", type=float, default=0.00015)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--workers", type=int, default="12")
    parser.add_argument("--data_parallel", action='store_true', default=False)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args

def parse_args_adni2_pet_mae_fine_tune() :
    parser = argparse.ArgumentParser()
    ### Dataset
    parser.add_argument("--path_dataset", type=str, default="/data/krdu/dataset")

    ### Model S/L
    parser.add_argument("--load_path", type=str, default="save_models_old")
    parser.add_argument("--load_model", action='store_true', default=True)
    parser.add_argument("--load_name", type=str, default="multi_B_mae")
    parser.add_argument("--save_path", type=str, default="saved_models")
    parser.add_argument("--model_name", type=str, default="vit_B")
    parser.add_argument("--model_type", type=str, default="vit_B", help="Which model to use (vit_L, vit_B, etc.)")

    ### Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--input_D", type=int, default=64)
    parser.add_argument("--input_H", type=int, default=64)
    parser.add_argument("--input_W", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--optimizer", type=str, default='AdamW')
    parser.add_argument("--scheduler", type=str, default='cosLR')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--workers", type=int, default="18")
    parser.add_argument("--data_parallel", action='store_true', default=False)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args