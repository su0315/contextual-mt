import pandas as pd
import numpy as np

df_train = pd.read_json("~/main/contextual-mt/data/BSD-master/train.json")
df_dev = pd.read_json("~/main/contextual-mt/data/BSD-master/dev.json")
df_test = pd.read_json("~/main/contextual-mt/data/BSD-master/test.json")

# Create Function for the rest
# make new df with id and mono-lingual sentence
# lang = [ja, en]
def new_format(data, lang):
    data.drop(['tag', 'title', 'original_language'], axis=1 )

    sent_list = []
    sent_no_list = []
    id_list = []


    for i in range(len(data['conversation'])):
        #print (len(conv))
        #print (df_train.index[df_train['conversation']==df_train['conversation'].iloc[i]])
        #df.index[df['column_name']==value].tolist()
        #conv_id = df_train['id'].iloc[conv.index]
        for sentence in data['conversation'].iloc[i]:
            id_list.append(data['id'].iloc[i])
            sent_no_list.append(sentence['no'])
            if lang == 'ja':
                sent_list.append(sentence['ja_sentence'])
            else:
                sent_list.append(sentence['en_sentence'])

    id_sentence_df = pd.concat([pd.DataFrame(id_list, columns = ['id']), pd.DataFrame(sent_no_list, columns = ['sentence_no']), pd.DataFrame(sent_list, columns = ['sentence'])],axis=1)
    id_sentence_df
    
    return id_sentence_df

ja_train=new_format(df_train, 'ja')
en_train = new_format(df_train, 'en')
ja_dev = new_format(df_dev, 'ja')
en_dev = new_format(df_dev, 'en')
ja_test = new_format(df_test, 'ja')
en_test = new_format(df_test, 'en')

# docid file generation, should be common in ja and en file
with open("/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/train.docids", 'w', encoding='utf-8') as wf:
    for line in en_train['id'].to_list(): 
        line = line.strip()
        wf.write(f'{line}\n')  

with open("/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/dev.docids", 'w', encoding='utf-8') as wf:
    for line in ja_dev['id'].to_list():
        line = line.strip()
        wf.write(f'{line}\n')  
        
with open("/home/sumire/main/contextual-mt/data/BSD-master/for_preprocess/test.docids", 'w', encoding='utf-8') as wf:
    for line in en_test['id'].to_list():
        line = line.strip()
        wf.write(f'{line}\n')  