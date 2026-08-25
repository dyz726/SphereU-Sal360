from __future__ import absolute_import, division, print_function

import math
import os
import argparse


parser = argparse.ArgumentParser(description="360 Degree Saliency Prediction Training")

                 
parser.add_argument("--num_workers", type=int, default=8, help="number of dataloader workers")

               
parser.add_argument("--task", type=str, default="salient", choices=["salient"])
parser.add_argument("--dataset_name", type=str, default="Sports-360",
                    choices=["Sports-360", "AVS-ODV", "SVGC_AVA"])
parser.add_argument(
    "--dataset_root_dir",
    type=str,
    default=None,
    help="dataset root containing training, testing, and videos; "
         "defaults to /home/dyz/PythonProject/Dataset/<dataset_name>",
)
parser.add_argument(
    "--avs_split",
    type=int,
    default=1,
    choices=[1, 2, 3],
    help="AVS-ODV 数据集划分方式编号，使用 train_list_N.txt / test_list_N.txt；"
         "仅在 --dataset_name 为 AVS-ODV 时生效",
)
parser.add_argument("--seq_length", type=int, default="30")         


                
parser.add_argument("--mode", type=str, default="vertex", choices=["face", "vertex"], help="folder to save the model in")              
parser.add_argument("--img_rank", type=int, default=6)            
parser.add_argument("--img_width", type=int, default=512)
parser.add_argument("--num_scales", type=int, default=4)             
parser.add_argument("--win_size_coef", type=int, default=2)            
parser.add_argument("--scale_factor", type=int, default=2)               
parser.add_argument("--abs_pos_enc_in", type=int, default=True)            
parser.add_argument("--abs_pos_enc", type=int, default=True)
parser.add_argument("--rel_pos_bias", type=int, default=True)              
parser.add_argument("--rel_pos_bias_size", type=int, default=7)             
parser.add_argument("--rel_pos_init_variance", type=float, default=1)                
parser.add_argument("--d_head_coef", type=int, default=2)                   
parser.add_argument("--enc_num_heads", nargs="+", type=int, default=[2,4,8,16])               
parser.add_argument("--dec_num_heads", nargs="+", type=int, default=[16,16,8,4])              
parser.add_argument("--bottleneck_num_heads", type=int, default=None)                                    
parser.add_argument("--scale_depth", type=int, default=2)                               
parser.add_argument("--debug_skip_attn", type=int, default=False)
parser.add_argument("--append_self", type=int, default=False)
parser.add_argument("--use_checkpoint", type=int, default=True)
parser.add_argument("--temporal_window_radius", type=int, default=5)                               

        
parser.add_argument("--dr", type=float, default=0.)
parser.add_argument("--dpr", type=float, default=0.)
parser.add_argument("--adr", type=float, default=0.)
parser.add_argument("--aodr", type=float, default=0.)
parser.add_argument("--posdr", type=float, default=0.)

                                                                            
                                                                         

parser.add_argument("--downsample", type=str, default="center")       
parser.add_argument("--upsample", type=str, default="interpolate")        

                       
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"], help="optimizer")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--min_learning_rate", type=float, default=1e-6, help="minimum learning rate for LR schedulers")
parser.add_argument(
    "--lr_scheduler",
    type=str,
    default="reduce_on_plateau",
    choices=["none", "warmup_cosine", "reduce_on_plateau"],
    help="learning rate scheduler",
)
parser.add_argument("--warmup_epochs", type=float, default=5, help="number of warmup epochs")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Adam weight decay")
parser.add_argument("--ltr", dest="limit_train_batches", type=int, default=math.inf, help="limit train batches per epoch")
parser.add_argument("--train_batch_size", type=int, default=1, help="batch size")
parser.add_argument("--val_batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--accum_grads", type=int, default=1, help="number of batches per optimizer update")
parser.add_argument("--base_model_weights", type=str,default="",help="预训练权重文件路径")

                              
parser.add_argument("--log_frequency", type=int, default=30, help="number of batches between each tensorboard models")
parser.add_argument("--tensorboard_log_dir", type=str, default=None,
                    help="TensorBoard log directory; defaults to <log_dir>/tensorboard")
parser.add_argument("--disable_tensorboard", dest="enable_tensorboard", action="store_false",
                    help="disable TensorBoard logging")
parser.add_argument("--enable_save", type=int, default=True, help="save model")
parser.add_argument("--save_frequency", type=int, default=1, help="number of epochs between each save")
parser.add_argument(
    "--load_weights_task",
    action="store_true",
    help="deprecated alias; requires --base_model_weights",
)

                            
parser.add_argument("--disable_color_augmentation",  dest="color_augmentation", action="store_false",
                    help="if set, do not use color augmentation")
parser.add_argument("--disable_lr_flip_augmentation", dest="lr_flip_augmentation", action="store_false",
                    help="if set, do not use left-right flipping augmentation")
parser.add_argument("--disable_yaw_rotation_augmentation", dest="yaw_rotation_augmentation", action="store_false",
                    help="if set, do not use yaw rotation augmentation")

                
parser.add_argument("--exp_name", default="train_sphereuformer", type=str)
parser.add_argument("--log_dir", default="log",type=str, help="models directory")
parser.add_argument("--wandb_entity", type=str)
parser.add_argument("--wandb_project", type=str)
parser.add_argument("--wandb_group", default=None, type=str)


parser.add_argument("--no_gpu", dest="use_gpu", action="store_false")
parser.add_argument("--test", action="store_true")

def main():
    args = parser.parse_args()

                                          
    if args.dataset_root_dir is None:
        args.dataset_root_dir = os.path.join(
            "/home/dyz/PythonProject/Dataset", args.dataset_name
        )

    if args.task == "depth":
        from trainer_dep import Trainer
    elif args.task == "segmentation":
        from trainer_seg import Trainer
    else:
        from train_salient import Trainer

    trainer = Trainer(args)

    if not args.test:
        trainer.train()
    else:
        trainer.test()


if __name__ == "__main__":
    main()

