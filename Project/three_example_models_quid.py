# -*- coding: utf-8 -*-
"""
https://quid.com/feed/how-quid-uses-deep-learning-with-small-data

Created on Sat Oct 21 15:40:02 2017

@author: Jannek

"""
from sklearn.pipeline import FeatureUnion,Pipeline

def plot_vectors(data):

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    ts = TSNE(2)
    reduced_vecs = ts.fit_transform(data)
    
    #color points by word group to see if Word2Vec can separate them
    for i in range(len(reduced_vecs)):
        if i < len(food_vecs):
            #food words colored blue
            color = 'b'
        elif i >= len(food_vecs) and i < (len(food_vecs) + len(sports_vecs)):
            #sports words colored red
            color = 'r'
        else:
            #weather words colored green
            color = 'g'
        plt.plot(reduced_vecs[i,0], reduced_vecs[i,1], marker='o', color=color, markersize=8)
    

# need to make own Transformer type classes for W2VExperiment, NewsldfStats, Punctuation, Capitalize

def get_features(): 
    
    tfidf = SklearnTfidf.from_file(os.path.join(TFIDF_FOLDER, '.pkl ' ),)
    word2vec_model = Word2Vec.load( os.path.join(W2V_FOLDER, 'model_bigram' ), ) 
    
    return FeatureUnion([
        # Using a pre—trained word2vec model on similar company data, 
        # we take the average word embedding, weighing contributions by tfidf 
        (' document _ vector (word2vec) ',W2VExperiment(word2vec_mode1, tfidf) ) , 
        #distributional information over word frequencies 
        ( ' idf stats', NewsldfStats(tfidf) ) , 
        #tfidf, over characters 
        ('tfidfl char_ngrams' , TfidfVectorizer( analyzer = 'char',ngram_range=(1, 4),lowercase=Fa1se) ), 
        #tfidf, over tokens
        (' tfidf token_ngrams' , TfidfVectorizer(ngram_range=(1, 4) ,lowercase=Fa1se) ) ,
        #distributional information over punctuation 
        (' punctuation' , Punctuation() ) , 
        #proportion of tokens that are capitalized 
        (' capitalize' , Capitalize() ) 
    ])
    
def get_model2(): 
    model = Pipeline( [ 
        ('feature_union' , get_features() ), 
        ('feature _ select' , SelectFpr(f_classif) ),
        ('Dense' , Densify() ),
#        using a custom sklearn extractor, we make predictions with 7 separate 
#        models, extract the probabilities, and then feed these into another 
#        logistic regression model. This way we can weigh more accurate models 
#        more heavily in the final prediction, and, through using polynomial 
#        features in a subsequent step, we can additionally act strategically 
#        based on situations where one classifier gives one prediction and another 
#        gives another type. 
        ('probas' ,ProbaExtractor( [ AdaBoostC1assifier( n_estimators=300), 
            ExtraTreesC1assifier(n_estimators=300) ,
            RandomForestC1assifier ( n_estimators=400 ), 
            LogisticRegression(), 
            BaggingC1assifier(), 
            KNeighborsC1assifier(), 
            GradientBoostingC1assifier()] ) ) ,     
        (' polynomial' , PolynomialFeatures (degree=2) ) , 
        (' classify' , LogisticRegression(C=0.5,random_state=666) ) 
        ])
    return model 

def get_model1():
    model = Pipeline([
        ('features',TfidVectorizer(ngram_range=(1,3),lowercase=True)),
        ('feature_select',SelectFpr(f_classif)),
        ('Dense',Densify()),
        ('logreg',GridSearchCV(LogisticRegressio(penalty='l2',param_grid)))
        ])
    return model

def get_model3(epoch, emb):

    graph_in = Input(shape=(emb.length, EMBEDDING_DIM))
    convs = []
    for fsz in range(1, 4):
        conv         = ConvolutionlD(nb_filter=300, filter_length=fsz,border_mode='valid', activation='relu',subsample_length=1)(graph_in)
        
        pool = MaxPoolinglD(pool_length=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)
    
    out = Merge(mode='concat')(convs)
    graph = Model(input=graph_in, output=out)
    model = Sequential()
    
    model.add(
        Embedding(max(emb.tokenizer.word_index.values()) + 1,
        EMBEDDING_DIM,
        weights=[emb.embeddings],
        input_length=emb.length,
        trainable=False)
        )
    
    model.add(graph)
    model.add(Dense(300))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy', fbeta_score])
    
    model.fit(emb.transform(epoch.X_train),
        epoch.y_train,
        validation_data=(emb.transform(epoch.X_test),
        epoch.y_test),
        nb_epoch=10,
        batch_size=16,
        callbacks = [
                TensorBoard(log_dir= './logs',
                            histogram_freq=0,
                            write_graph=True,
                            write_images=False)
        ])

return model


