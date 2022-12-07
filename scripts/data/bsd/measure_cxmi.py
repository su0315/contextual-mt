from contextual_mt.utils import parse_documents
from contextual_mt.docmt_cxmi import compute_cxmi
import numpy as np

context_model_ckpt = "/home/sumire/main/NMT_models/jparacrawl/en-ja/context_model_ckpt"
source_context_size=0
target_context_size=0
source_lang="en"
target_lang="ja"

source_file="/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/test.en"
target_file="/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/test.ja"
docids_file="/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/.docids"
batch_size=8


