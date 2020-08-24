import os 


def run(step=0):
    if step == 0:
        # pre process data
        os.system('onmt_preprocess -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo')
    if step == 1:
        os.system('onmt_train -data data/demo -save_model demo-model')
    if step == 2:
        os.system('onmt_translate -model demo-model_XYZ.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose')


if __name__ == '__main__':
    run()

