

def getData():
    import numpy as np
    import pandas as pd
    # load input feature dataset for Agre
    Agre_asd = pd.read_csv("data/v34_lof_asd_af0.50.txt", index_col=0).transpose()
    Agre_ctrl = pd.read_csv("data/v34_lof_typical_af0.50.txt", index_col=0).transpose()

    print("Cases: ", Agre_asd.shape[0])
    print("Controls: ", Agre_ctrl.shape[0])

    # load input feature dataset for SSC
    SSC_asd = pd.read_csv("data/SSC_lof_asd_af0.50.txt", index_col=0).transpose()
    SSC_ctrl = pd.read_csv("data/SSC_lof_typical_af0.50.txt", index_col=0).transpose()

    print ("Cases: ", SSC_asd.shape[0])
    print ("Controls: ", SSC_ctrl.shape[0])


    # merge SSC and Agre data
    X_asd = pd.concat([SSC_asd, Agre_asd], axis = 0).fillna(0)
    X_ctrl = pd.concat([SSC_ctrl, Agre_ctrl], axis = 0).fillna(0)

    X = pd.concat([X_asd, X_ctrl], axis=0)
    print ("Total cases: ", X_asd.shape[0])
    print ("Total controls: ", X_ctrl.shape[0])
    print ("Features (ie. genes): ", X.shape[1])
    print ("Missing Values: ", int(X.isnull().values.any()))

    ### Target Data (ASD/non-ASD diagnosis)

    #We have a file that Kelley has made with inferred Autism/Control diagnosis for the individuals in the iHart study.  We will try and predict #diagnosis 0 = Control, 1 = Austism.

    y = pd.read_csv("data/all_samples_filtered_labels.csv", usecols = ['identifier','diagnosis'], index_col=0)
    # shift y to a 0/1 representation for Control/ASD
    y["diagnosis"] = np.where(y['diagnosis'] == 'Autism', 1, 0)

    # get lists of individuals in X and Y
    m_x = X.index.values.tolist()
    m_x_asd = X_asd.index.tolist()
    m_x_ctrl = X_ctrl.index.tolist()
    m_y = y.index.values.tolist()

    # check subject overlap between X and Y
    print ("%d subjects in X are not in y.  Of these, %d are cases and %d are controls." % (len(set(m_x) - set(m_y)), len(set(m_x_asd) - set(m_y)), len(set(m_x_ctrl) - set(m_y))))

    # make a list of Subject IDs with overlap
    subjects = list(set(m_x) & set(m_y))
    print ("This leaves %d subjects: %d cases and %d controls." % (len(subjects), len(set(m_x_asd) & set(m_y)), len(set(m_x_ctrl)&set(m_y))) )



    #**Note:** The set of "cases" and "controls" appear to be differently defined between the iHart Phenotype labels (i.e. our `y` labels) and the CGT matrix labels (i.e. our `X` features). 

    #You can notice that the majority of controls don't appear in our phenotype information dataset. This is because ADOS\ADI-R was not administered to many controls from SSC and Agre. Since we're interested in classifying ASD/non-ASD, for our purposes it is not necessary to exclude these individuals because we do not necessarily need any phenotype information outside of diagnosis. Rather, we can infer that all individuals in a 'control' CGT matrix without ADOS/ADI-R information have a non-ASD diagnosis.

    to_add = list(set(m_x_ctrl) - set(m_y))
    y_ctrl = pd.DataFrame(np.zeros(len(to_add),), columns = ['diagnosis'],index = to_add)
    y = pd.concat([y, y_ctrl], axis = 0)
    subjects = subjects + to_add
    print (len(subjects))
    print (y.shape)


    # redefine X and Y to contain only the subjects we want
    X = X.ix[subjects]
    y = y.ix[subjects]

    # check we have the same subject IDs in the same order for X and Y
    print( y.index.values.tolist() == X.index.values.tolist())
    y = y.ix[:,0]
    print (y.value_counts())

    #One thing that's probably going to be an issue for this experiment is that there are very few controls for whom we have both genetic and ADOS/ADI-R information.  This is going to mean that a random classifier performs with fairly high accuracy, just because classifying most or all individuals as autistic is a effective strategy when we have so few negatives.

    random.seed(143)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return(X_train, X_test, y_train, y_test)