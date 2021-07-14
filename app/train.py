import os
import pandas as pd

from fastai.text import *
import multifit
import argparse

NUM_EPOCHS = 16
SEED = 10
# batch_size = exp.finetune_lm.bs
batch_size = 64

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(args):
    exp = multifit.from_pretrained(args.pretrained_model)
    NUM_EPOCHS = args.epochs
    batch_size = args.batch_size
    SEED = args.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible

    path = os.getcwd()
    data_dir = os.path.join(path, args.data_dir)
    cls_dataset = exp.arch.dataset(Path(data_dir), exp.pretrain_lm.tokenizer)
    data_clas = cls_dataset.load_clas_databunch(bs=batch_size)

    # train
    exp.finetune_lm.train_(cls_dataset, num_epochs=NUM_EPOCHS)
    exp.load_(cls_dataset.cache_path/exp.pretrain_lm.name)

    for seed in [SEED]:
        exp.classifier.train_(seed=seed, num_epochs=NUM_EPOCHS)

    # test
    def get_results(exp_path):
        exp = multifit.ULMFiT().load_(exp_path, silent=True).classifier
        results = exp.validate(use_cache=True)
        results.update(seed=exp.seed, fp16=exp.fp16)
        return results

    results = [get_results(exp_path) for exp_path in cls_dataset.cache_path.glob(exp.pretrain_lm.name+"seed*")]
    results_df = pd.DataFrame.from_records(results)
    results_df.sort_values(["valid accuracy"])[["name", "seed", "test accuracy", "valid accuracy"]]
    print(results_df)
    results_df.to_csv('./train_result.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pretrained_model', default='ja_multifit_paper_version', type=str,
                        help='pretrained model name')
    parser.add_argument('--data_dir', default='./data', type=str,
                        help='folder contains sample_train.csv, test.csv (and optional dev.csv)')
    parser.add_argument('--cuda_visible', default='', type=str,
                        help='GPU for training')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of training epochs.')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for shuffling training data')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size for training')

    args = parser.parse_args()
    train(args)
