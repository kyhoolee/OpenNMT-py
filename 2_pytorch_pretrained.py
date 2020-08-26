import torch
import fairseq

def run():
    # List available models
    avail_model = torch.hub.list('pytorch/fairseq')  # [..., 'transformer.wmt16.en-de', ... ]
    print('AVAIL-MODELS:: ', '\n'.join(avail_model))

    # Load a transformer trained on WMT'16 En-De
    # Note: WMT'19 models use fastBPE instead of subword_nmt, see instructions below
    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model',
                        tokenizer='moses', bpe='fastbpe')
    en2de.eval()  # disable dropout

    # The underlying model is available under the *models* attribute
    assert isinstance(en2de.models[0], fairseq.models.transformer.TransformerModel)

    # Move model to GPU for faster translation
    # en2de.cuda()

    # Translate a sentence
    # en2de.translate('Hello world!')
    # 'Hallo Welt!'

    # Batched translation
    result = en2de.translate(['Miranda Kerr and Orlando Bloom are parents to two-year-old Flynn .'])
    # (['Hello world!', 'The cat sat on the mat.'])
    print('RESULT:: ',result)
    # ['Hallo Welt!', 'Die Katze saß auf der Matte.']


def run_fr():
    # Load an En-Fr Transformer model trained on WMT'14 data :
    en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt')

    # Use the GPU (optional):
    # en2fr.cuda()

    # Translate with beam search:
    fr = en2fr.translate('Hello world!', beam=5)
    assert fr == 'Bonjour à tous !'
    print(fr)

    # Manually tokenize:
    en_toks = en2fr.tokenize('Hello world!')
    assert en_toks == 'Hello world !'
    print(en_toks)

    # Manually apply BPE:
    en_bpe = en2fr.apply_bpe(en_toks)
    assert en_bpe == 'H@@ ello world !'
    print(en_bpe)

    # Manually binarize:
    en_bin = en2fr.binarize(en_bpe)
    assert en_bin.tolist() == [329, 14044, 682, 812, 2]
    print(en_bin)

    # Generate five translations with top-k sampling:
    fr_bin = en2fr.generate(en_bin, beam=5, sampling=True, sampling_topk=20)
    assert len(fr_bin) == 5
    print(fr_bin)

    print('----------------------')
    # Convert one of the samples to a string and detokenize
    fr_sample = fr_bin[0]['tokens']
    print('fr_sample: ', fr_sample)
    fr_bpe = en2fr.string(fr_sample)
    print('fr_bpe: ', fr_bpe)
    fr_toks = en2fr.remove_bpe(fr_bpe)
    print('fr_toks: ', fr_toks)
    fr = en2fr.detokenize(fr_toks)
    print('fr: ', fr)
    assert fr == en2fr.decode(fr_sample)


if __name__ == '__main__':
    run()