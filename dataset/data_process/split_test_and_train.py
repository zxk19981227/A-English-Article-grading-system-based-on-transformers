import pickle
import random
import numpy as np
score=pickle.load(open('./drive/My Drive/data/Score.ds','rb'))
group=pickle.load(open('./drive/My Drive/data/Group.ds','rb'))
article=pickle.load(open('./drive/My Drive/data/Article.ds','rb'))

score=score.tolist()
group=group.tolist()
article=article.tolist()
test_score=[]
test_group=[]
test_article=[]
test_address=[]
train_address=[]
train_score=[]
train_article=[]
train_score_address=[]
test_score_address=[]
for i in range(8):
    train_score.append([])
    test_score.append([])
    test_group.append([])
    test_article.append([])
    test_address.append('./drive/My Drive/data/group_test_and_train/test_'+str(i+1)+'_article.ds')
    train_address.append('./drive/My Drive/data/group_test_and_train/train_'+str(i+1)+'_article.ds')
    train_score_address.append('./drive/My Drive/data/group_test_and_train/train_'+str(i+1)+'_score.ds')
    test_score_address.append('./drive/My Drive/data/group_test_and_train/test_'+str(i+1)+'_score.ds')
for num,(sco,gro,arti) in enumerate(zip(score,group,article)):
    if(num%5==0):
        test_score[gro-1].append(sco)
        #test_group[].append(gro)
        test_article[gro-1].append(arti)
        score.remove(sco)
        group.remove(gro)
        article.remove(arti)
    else:
        train_score[gro-1].append(sco)
        train_article[gro-1].append(arti)
        score.remove(sco)
        group.remove(gro)
        article.remove(arti)
for i in range (8):
    with open(train_address[i]) as f:
        pickle.dump(train_article[i],f)
    with open(train_score_address[i]) as f:
        pickle.dump(train_score[i],f)
    with open(test_address[i]) as f:
        pickle.dump(test_article[i],f)
    with open(test_score_address[i]) as f:
        pickle.dump(test_score[i],f)

