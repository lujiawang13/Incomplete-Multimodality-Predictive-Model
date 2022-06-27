import numpy as np
import math

from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error


def dict_list(Adict, keynames):
    values = []
    for keyname in keynames:
        value = Adict[keyname]
        if isinstance(value, list) == False:
            if len(value.shape) == 1:
                value = value.tolist()
        values += value
    return values

def groupsdataframe_f(groupsdata,groupsnames_,feature):
    groupsdataframe=dict()
    for groupname in groupsnames_:
        groupsdataframe[groupname]=groupsdata[groupname].loc[:,feature]
    return groupsdataframe

class beta_f():
    # init method or constructor
    def __init__(self, TrainData, blocktypes,Yname, groupscompletedata,
                 groupsnames_,blocks_features,groups_alpha):
        self.TrainData = TrainData
        self.blocktypes = blocktypes
        self.Yname = Yname
        self.groupscompletedata = groupscompletedata
        self.groupsnames_=groupsnames_
        self.blocks_features=blocks_features
        self.groups_alpha=groups_alpha
        self.allfeatures = dict_list(blocks_features, blocktypes)
        self.alpha=np.array(dict_list(groups_alpha, groupsnames_))
        self.n=TrainData.shape[0]


    # computer beta when alpha is fixed
    '''
      one group: x1,x2; another group: x2,x3
      ypred=alpha11x1beta1+alpha12x2beta2+alpha13(x3=0)beta3+
      alpha21(x1=0)beta1+alpha22x2beta2+alpha23x3beta3
      z=[x1beta1,x2beta2,x3beta3]
      loss=(z1alpha1-y1)^2+(z2alpha2-y2)^2
      bigZ=[[z1,0],[0,z2]]
      y=[y1,y2]
      alpha=[alpha1,alpha2]
    '''

    def onegroups_z_f(self,onegroupcompletedata,onegroup_alpha):
        n_groupi = onegroupcompletedata.shape[0]
        i = 0
        alphax = np.array([]).reshape(n_groupi, 0)
        for blockname in self.blocktypes:
            onegroupcompletedata_block = onegroupcompletedata.loc[:, self.blocks_features[blockname]]
            onealphax = onegroupcompletedata_block.to_numpy() * np.array(onegroup_alpha[i])
            alphax = np.concatenate((alphax,
                                     onealphax), axis=1)
            i = i + 1
        return alphax


    def groups_z_f(self):
        groups_z=dict()
        for groupname_ in self.groupsnames_:
            onegroupcompletedata=self.groupscompletedata[groupname_]
            onegroup_alpha=self.groups_alpha[groupname_]
            alphax=self.onegroups_z_f(onegroupcompletedata, onegroup_alpha)

            groups_z[groupname_]=alphax
        return groups_z
    def Zbig_f(self):
        groups_z=self.groups_z_f()
        alphax=groups_z[self.groupsnames_[0]]
        Zbig = alphax*math.sqrt(self.n/alphax.shape[0])
        for groupname in self.groupsnames_[1:len(self.groupsnames_)]:
            alphax = groups_z[groupname]
            group_z=alphax*math.sqrt(self.n/alphax.shape[0])
            Zbig=np.concatenate((Zbig,group_z), axis=0)
        return Zbig

    def blockscoef_f(self,coefs):
        i = 0
        blocks_coef = dict()
        for blocktype in self.blocktypes:
            blockfeatures_num=len(self.blocks_features[blocktype])
            blocks_coef[blocktype] = coefs[i:(i + blockfeatures_num)]
            i = i + blockfeatures_num
        return blocks_coef

    def ydict_list(self,Adict, keynames):
        values = []
        for keyname in keynames:
            value = Adict[keyname]
            value = value.to_numpy()
            value = value * math.sqrt(self.n / len(value))
            values += list(value)
        return values

    def betaupdate_f(self,lamda,coefspenalty,fit_intercept):
        Zbig=self.Zbig_f()
        X=Zbig
        groups_y=groupsdataframe_f(self.groupscompletedata, self.groupsnames_, self.Yname)
        y=self.ydict_list(groups_y, self.groupsnames_)

        if coefspenalty=='ridge':
            model = Ridge(normalize = False,fit_intercept=fit_intercept)
            #################################
            # model = Ridge(alpha=lamda, normalize=False)
            # model.fit(X,y)
        elif coefspenalty=='LASSO':
            model = Lasso(max_iter=10000, normalize=False,fit_intercept=fit_intercept)


        model.set_params(alpha=lamda)
        model.fit(X, y)
        beta = model.coef_

        y_pred = model.predict(X)

        MSE = sum((y_pred - y) ** 2)
        #MSE = sum(abs(y_pred - y) )
        # MSE= mean_squared_error(y, y_pred) # squared = False

        if coefspenalty == 'ridge':
            penalty=sum(beta**2)
        elif coefspenalty == 'LASSO':
            penalty = sum(abs(beta))

        objl=(MSE+penalty*lamda)/len(y)


        blocks_beta=self.blockscoef_f(beta)

        self.intercept_=model.intercept_
        self.model=model
        return beta,blocks_beta,objl

