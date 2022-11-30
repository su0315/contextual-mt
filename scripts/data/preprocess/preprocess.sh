VOCAB_SIZE=32000

REPO=C:/Users/hsmr0/IM/main/contextual-mt
data_dir=C:/Users/hsmr0/IM/main/contextual-mt/data/BSD-master/for_preprocess
nmt_models_dir=C:/Users/hsmr0/IM/main/NMT_models

for lang in $src_lang $tgt_lang; do
    python $REPO/scripts/spm_train.py \
        ${data_dir}/train.${lang} \
        --model-prefix ${data_dir}/prep/spm.${lang}.nopretok.model \ 
        --vocab-file ${nmt_models_dir}/jparacrawl/small_${src_lang}-${tgt_lang}/dict.${lang}.txt \
        --vocab-size $VOCAB_SIZE
done
for split in train dev test; do
    for lang in $src_lang $tgt_lang; do
        python $REPO/scripts/spm_encode.py \
            --model ${data_dir}/prep/spm.$lang.nopretok.model \
                < ${data_dir}/${split}.${lang} \
                > ${data_dir}/prep/${split}.sp.${lang}
    done
done

src_lang="en"
tgt_lang="ja"

fairseq-preprocess \
    --source-lang $src_lang --target-lang $tgt_lang \
    --trainpref ${data_dir}/prep/train.sp \
    --validpref ${data_dir}/pep/valid.sp \
    --testpref ${data_dir}/prep/test.sp \
    --srcdict ${nmt_models_dir}/jparacrawl/${src_lang}-${tgt_lang}/small_${src_lang}-${tgt_lang}/dict.${src_lang}.txt 
    --tgtdict ${nmt_models_dir}/jparacrawl/${src_lang}-${tgt_lang}/small_${src_lang}-${tgt_lang}/dict.${tgt_lang}.txt
    --destdir ${data_dir}/bin

