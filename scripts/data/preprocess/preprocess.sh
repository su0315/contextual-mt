VOCAB_SIZE=32000

REPO=C:/Users/hsmr0/IM/main/contextual-mt
data_dir=C:/Users/hsmr0/IM/main/contextual-mt/data/BSD-master/for_preprocess

echo $REPO


for lang in ja en; do
    python $REPO/scripts/spm_train.py \
        ${data_dir}/train.${lang} \
        --model-prefix ${data_dir}/prep/spm.${lang}.nopretok \ 
        --vocab-file ${data_dir}/prep/dict.${lang}.txt \
        --vocab-size $VOCAB_SIZE
done
for split in train dev test; do
    for lang in ja en; do
        python $REPO/scripts/spm_encode.py \
            --model ${data_dir}/prep/spm.$lang.nopretok.model \
                < ${data_dir}/${split}.${lang} \
                > ${data_dir}/prep/${split}.sp.${lang}
    done
done

fairseq-preprocess \
    --source-lang src --target-lang tgt \
    --trainpref ${data_dir}/prep/train.sp \
    --validpref ${data_dir}/pep/valid.sp \
    --testpref ${data_dir}/prep/test.sp \
    --srcdict ${data_dir}/prep/dict.src.txt \
    --tgtdict ${data_dir}/prep/dict.tgt.txt \
    --destdir ${data_dir}/bin

