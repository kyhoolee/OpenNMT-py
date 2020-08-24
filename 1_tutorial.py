import os 
import sys
import argparse


def run(step=0):
    if step == 0:
        # pre process data
        print('STEP-0: PREPROCESS DATA')
        os.system('onmt_preprocess -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/tut1')
    elif step == 1:
        print('STEP-0: TRAIN MODEL')
        os.system('onmt_train -data data/tut1 -save_model tut1_model')
    elif step == 2:
        print('STEP-0: PREPROCESS DATA')
        os.system('onmt_translate -model demo-model_XYZ.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0, choices=[0, 1, 2],
                        help='specify which step to run')

    args = parser.parse_args()
    
    run(args.step)

