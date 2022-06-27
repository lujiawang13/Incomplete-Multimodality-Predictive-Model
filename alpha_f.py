import numpy as np
from sklearn.linear_model import Ridge, Lasso


def dict_list(Adict, keynames):
    values = []
    for keyname in keynames:
        value=Adict[keyname]
        if isinstance(value,list)==False:
            if len(value.shape)==1:
                value=value.tolist()
        values += value
    return values

    # for k, v in groupdata.items():
    #     print(k)

def groupsdataframe_f(groupsdata,groupsnames_,feature):
    groupsdataframe=dict()
    for groupname in groupsnames_:
        groupsdataframe[groupname]=groupsdata[groupname].loc[:,feature]
    return groupsdataframe
class alpha_f():
    # init method or constructor
    def __init__(self, TrainData, blocktypes,Yname, groupscompletedata,
                 groupsnames_,blocks_features,blocks_beta):
        self.TrainData = TrainData
        self.blocktypes = blocktypes
        self.Yname = Yname
        self.groupscompletedata = groupscompletedata
        self.groupsnames_=groupsnames_
        self.blocks_features=blocks_features
        self.blocks_beta=blocks_beta
        self.allfeatures = dict_list(blocks_features, blocktypes)
        self.beta=np.array(dict_list(blocks_beta, blocktypes))


    # computer alpha when beta is fixed
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
    def groups_z_f(self):
        groups_z=dict()
        for groupname in self.groupsnames_:
            xbeta=np.array([]).reshape(self.groupscompletedata[groupname].shape[0],0)
            for blockname in self.blocktypes:
                onegroupcompletedata=self.groupscompletedata[groupname].loc[:,self.blocks_features[blockname]]
                onexbeta=np.dot(onegroupcompletedata.to_numpy(),np.array(self.blocks_beta[blockname]))
                xbeta=np.concatenate((xbeta,
                                      onexbeta.reshape(len(onexbeta),1)), axis=1)
            groups_z[groupname]=xbeta
        return groups_z
    def Zbig_f(self):
        groups_z=self.groups_z_f()
        Zbig = groups_z[self.groupsnames_[0]]
        for groupname in self.groupsnames_[1:len(self.groupsnames_)]:
            group_z=groups_z[groupname]
            zerosupper=np.zeros((Zbig.shape[0],group_z.shape[1]))
            Zbigzero=np.concatenate((Zbig, zerosupper), axis=1)
            zeroslow = np.zeros((group_z.shape[0], Zbig.shape[1]))
            Zsmallzeeo=np.concatenate((zeroslow, group_z), axis=1)
            Zbig=np.concatenate((Zbigzero, Zsmallzeeo), axis=0)
        return Zbig

    def groupscoef_f(self,coefs):
        S=len(self.blocktypes)
        i = 0
        groups_coef = dict()
        for groupname in self.groupsnames_:
            groups_coef[groupname] = coefs[i:(i + S)]
            i = i + S
        return groups_coef

    def alphaupdate_f(self,mu,blockwisepenalty):
        Zbig=self.Zbig_f()
        X=Zbig
        groups_y=groupsdataframe_f(self.groupscompletedata, self.groupsnames_, self.Yname)
        y=dict_list(groups_y, self.groupsnames_)

        if blockwisepenalty=='ridge':
            ridge = Ridge(normalize = False,fit_intercept=False)
            ridge.set_params(alpha=mu)
            ridge.fit(X, y)
            alpha=ridge.coef_

            #################################
            # ridge = Ridge(alpha=mu, normalize=False)
            # ridge.fit(X,y)
            # alpha=ridge.coef_
        elif blockwisepenalty=='LASSO':
            lasso = Lasso(max_iter=10000, normalize=False,fit_intercept=False)
            lasso.set_params(alpha=mu)
            lasso.fit(X, y)
            alpha=lasso.coef_

        groups_alpha=self.groupscoef_f(alpha)
        return alpha,groups_alpha