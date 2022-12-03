#!/bin/bash

VOCAB_SIZE=32000

REPO=/home/sumire/main/contextual-mt
data_dir=/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess
nmt_models_dir=/home/sumire/main/NMT_models

src_lang="en"
tgt_lang="ja"

for lang in $src_lang $tgt_lang; do
    python $REPO/scripts/spm_train.py \
        ${data_dir}/train.${lang} \
        --model-prefix ${data_dir}/prep/spm.${lang}.nopretok.model \
        --vocab-file ${nmt_models_dir}/jparacrawl/${src_lang}-${tgt_lang}/small_${src_lang}-${tgt_lang}/dict.${lang}.txt \
        --vocab-size $VOCAB_SIZE
done
for split in train dev test; do
    for lang in $src_lang $tgt_lang; do
        python $REPO/scripts/spm_encode.py \
            --model ${data_dir}/prep/spm.$lang.nopretok.model \
            --inputs ${data_dir}/${split}.${lang} \
            --outputs ${data_dir}/prep/${split}.sp.${lang} # with open('file', 'w') as sys.stdout: print('test')

    done

done

fairseq-preprocess \
    --source-lang $src_lang --target-lang $tgt_lang \
    --trainpref ${data_dir}/prep/train.sp \
    --validpref ${data_dir}/pep/valid.sp \
    --testpref ${data_dir}/prep/test.sp \
    --srcdict ${nmt_models_dir}/jparacrawl/${src_lang}-${tgt_lang}/small_${src_lang}-${tgt_lang}/dict.${src_lang}.txt \
    --tgtdict ${nmt_models_dir}/jparacrawl/${src_lang}-${tgt_lang}/small_${src_lang}-${tgt_lang}/dict.${tgt_lang}.txt \
    --destdir ${data_dir}/bin

