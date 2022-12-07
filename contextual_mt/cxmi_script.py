# train file
#source_file="/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/test.en"
#target_file="/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/test.ja"
#docids_file="/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/test.docids"
#batch_size=8


docmt_cxmi.py --source-file "/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/test.en" --docids-file "/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/test.docids" --reference-file "/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/test.ja" --path "home/sumire/main/NMT_models/jparacrawl/en-ja/small_en-ja"
#small.pretrain.pt