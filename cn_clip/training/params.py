import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B-32", "ViT-B-16"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    elif model_name in ["ViT-L-14"]:
        return {"lr": 4.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def parse_args():
    parser = argparse.ArgumentParser()
    # --------------------------- 训练/验证数据 ------------------------------------------
    parser.add_argument(
        "--train-data",
        type=str,
        # required=True,
        default=r'E:\common_tools\Chinese-CLIP-master\dataset\furnitures\train',
        help="训练数据LMDB目录", )  # 其中训练数据tsv：1000002	/9j/4AAQSkZJ...YQj7314oA//2Q==，
    # 记录的是图像信息，jsonl：{"text_id": 8428, "text": "高级感托特包斜挎", "image_ids": [1076345, 517602]}，记录是文本信息及图文对匹配关系
    parser.add_argument(
        "--val-data",
        type=str,
        # default=None,
        default=r'E:\common_tools\Chinese-CLIP-master\dataset\furnitures\train',
        help="验证数据LMDB目录，默认是None", )  # 可以用28w做训练数据，1w4做验证数据
    parser.add_argument(
        "--num-workers", type=int, default=0, help="训练数据处理(Dataloader)的进程数，默认是4")

    # -------------------------------- 训练超参数 -------------------------------------
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14"],
        default="ViT-B-16",
        help="指定视觉backbone，从 [‘ViT-B-32’, ‘ViT-B-16’, ‘ViT-L-14’]选择", )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="指定文本backbone, 从 [‘RoBERTa-wwm-ext-base-chinese’, ‘RoBERTa-wwm-ext-large-chinese’]选择", )  # 文本模型的用科大讯飞
    parser.add_argument(
        "--context-length", type=int,
        # default=64,
        default=24,
        help="文本输入序列长度 (include [CLS] & [SEP] tokens).")
    parser.add_argument(
        "--warmup", type=int,
        # default=500,
        default=100,
        help="warmup步数")
    parser.add_argument(
        "--batch-size", type=int, default=24,
        help="训练时单卡的bs，保证训练样本总数>bs*GPU数")  # 推荐使用更大的bs，contrastive learning很考虑正负样本数量，否则微调lr和训练过程。
    parser.add_argument("--lr", type=float,
                        # default=None,
                        default=5e-5,
                        help="Learning rate.")
    parser.add_argument("--wd", type=float,
                        # default=0.2,
                        default=0.001,
                        help="Weight decay.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--use-bn-sync",
                        default=False,
                        action="store_true",
                        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.")
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="训练步数，也可以通过max-epochs指定训练轮数，Number of steps to train for (in higher priority to --max_epochs).")
    parser.add_argument(
        "--max-epochs", type=int, default=32, help="Number of full epochs to train for (only works if --max_steps is None).")

    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        default=False,
        help="是否freeze视觉backbone", )  # wukong是fix住image tower的权重，chineseclip是图像和文本两个都fine，若从头训练，text侧很难收敛。
    parser.add_argument(
        "--clip-weight-path",
        default=None,
        type=str,
        help="The path of openai pretrained weight, used to initialize the image encoder, should be set to None if you do not use pretrained CLIP", )
    parser.add_argument(
        "--bert-weight-path",
        default=None,
        type=str,
        help="The path of bert pretrained weight, used to initialize the text encoder, should be set to None if you do not use pretrained BERT", )

    parser.add_argument("--use-augment",
                        # default=False,
                        default='--use-augment',
                        action="store_true",
                        help="是否使用AutoAugment对图片进行数据增强")
    parser.add_argument(
        "--valid-batch-size", type=int, default=64, help="验证时单机bs，保证验证集样本总数>bs*GPU数")
    parser.add_argument(
        "--valid-step-interval", type=int,
        # default=None,
        default=150,
        help="验证step频率，指定-1时则在训练中不进行验证(default to None which disables validation between steps).")
    parser.add_argument(
        "--valid-epoch-interval", type=int, default=1,
        help="验证epoch频率，指定-1时则在训练中不进行验证(default to 1, set None to disable validation between epochs).")

    # ------------------------------------------ 输出选项 ---------------------------------
    parser.add_argument(
        "--logs",
        type=str,
        default=r"F:\gitlab\material\clip\results",
        help="存储log的目录", )
    parser.add_argument(
        "--name",
        type=str,
        default="train_clip_furniters",
        help="指定输出路径。超参日志, 训练日志以及产出ckpt均会存放至. Otherwise use current time.", )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="How often to log loss info.")
    parser.add_argument(
        "--save-epoch-frequency", type=int, default=1, help="How often to save checkpoints by epochs.")  # 均以epoch为主
    parser.add_argument(
        "--save-step-frequency", type=int, default=-1, help="How often to save checkpoints by steps.")

    # -------------------------- 权重读取相关选项 ------------------------------------------
    parser.add_argument(
        "--resume",
        # default=None, # 指定为None，则从头训练
        default=r"F:\gitlab\material\clip\weights\clip_cn_vit-b-16.pt",
        type=str,
        help="权重读取的路径。示例脚本中指定为预训练ckpt路径，也可以指定为用户自己finetune的ckpt路径做继续训练。", )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        # default=False,
        default="--reset-optimizer",
        help="是否使用optimizer state，If resumed from a checkpoint, whether to reset the optimizer states.", )
    parser.add_argument(
        "--reset-data-offset",
        action="store_true",
        # default=False,
        default="--reset-data-offset",
        help="是否从此前的数据断点续跑。如batch size或GPU卡数超参改变，建议打开此选项。If resumed from a checkpoint, whether to reset the dataset offset to the beginning.", )
    parser.add_argument(
        "--report-training-batch-acc",
        # default=False,
        default='--report-training-batch-acc',
        action="store_true", help="Whether to report training batch accuracy.")
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition.")
    # arguments for distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank.")
    parser.add_argument(
        "--skip-aggregate",
        default=False,
        action="store_true",
        help="whether to aggregate features across gpus before computing the loss")
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.")
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed.")
    args = parser.parse_args()
    args.aggregate = not args.skip_aggregate

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.vision_model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
