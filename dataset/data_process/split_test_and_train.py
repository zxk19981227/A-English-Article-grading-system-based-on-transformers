import random
"""
import re
UNKNOWN_TOKEN = "UNK"

DROPOUT_TOKENS = {"a", "an", "the", "'ll", "'s", "'m", "'ve"}  # Add "to"
SHORT_TOKENS={"ll", "s", "m", "ve"}
REPLACEMENTS = {"there": "their", "their": "there", "then": "than",
                "than": "then"}
dropout_prob=0.25
replacement_prob=0.25

# Add: "be":"to"
with open("../../dataset/translate-dataset/train_set.txt",'w') as out_f:
    with open("../../dataset/translate-dataset/movie_dialog_train.txt",'r') as f:
        for each_line in f.readlines():
            each_line=each_line.lower().strip().split()
            source = []
            target = []

            for token in each_line:
                target.append(token)

              # Randomly dropout some words from the input.
                dropout_token = (token in DROPOUT_TOKENS and
                             random.random() < dropout_prob)
                replace_token = (token in REPLACEMENTS and
                             random.random() < replacement_prob)

                if replace_token:
                    source.append(REPLACEMENTS[token])
                elif not dropout_token:
                    source.append(token)
            st=""
            for e in source:
                if(st==""):
                    st=e
                else:
                    st=st+" "+e
            out_f.writelines(st)
            out_f.writelines('\n')

"""
val_set=open("../../dataset/translate-dataset/val_set.txt",'w')
train_set=open("../../dataset/translate-dataset/training_set.txt",'w')

with open("../../dataset/translate-dataset/train_set.txt",'r') as out_f:
    with open("../../dataset/translate-dataset/movie_dialog_train.txt",'r') as f:
        source=out_f.readlines()
        target=f.readlines()
        for sou,tar in zip(source,target):
            if(random.random()<0.1):
                val_set.writelines("src:"+sou)
                val_set.writelines("tar:"+tar)
            else:
                train_set.writelines("src:" + sou )
                train_set.writelines("tar:" + tar)


