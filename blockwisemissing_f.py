import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

from Datapreprocessing_f import Datapreprocessing_f,blocksfeatures_f
from alpha_f import alpha_f
from beta_f import beta_f


def Scale_f(Data,allfeatures,normalize):
    if normalize==False:
        return
    xData = Data.loc[:, allfeatures]

    scaler = StandardScaler()
    scaler.fit(xData)
    return scaler


def ScaleData_f(Data, allfeatures,scaler):
    if scaler:
        xData = Data.loc[:, allfeatures]

        xData_scale = scaler.transform(xData)
        xData_scale = pd.DataFrame(xData_scale, columns=allfeatures,index=Data.index)

        otherData = Data.drop(columns=allfeatures)

        scaleData = pd.concat([otherData, xData_scale], axis=1)
        #print({'scaleTestData': scaleData.shape})
        return scaleData
    else:
        return Data


class blockwisemissing_f():
    # init method or constructor
    def __init__(self, TrainData, blocktypes,Yname,lamda,coefspenalty,mu,blockwisepenalty,normalize):
        #self.TrainData = TrainData
        self.blocktypes = blocktypes
        self.Yname=Yname
        self.lamda=lamda
        self.coefspenalty=coefspenalty
        self.mu=mu
        self.blockwisepenalty=blockwisepenalty
        self.normalize=normalize

        self.allfeatures,self.blocks_features=blocksfeatures_f(TrainData, self.blocktypes)


        # scaling
        self.scaler = Scale_f(TrainData, self.allfeatures,self.normalize)
        self.TrainData=ScaleData_f(TrainData, self.allfeatures, self.scaler)

        Datapreprocessing = Datapreprocessing_f(self.TrainData, self.blocktypes, self.Yname)

        groupsdata, groupscompletedata, groupsnames_ = Datapreprocessing.groupsdata_f()

        # S: number of blocks
        S = len(self.blocktypes)
        # all possible groups: groupnum=2**S-1; In practice, it is less
        groupnum = len(groupsdata)

        self.groupsdata=groupsdata
        self.groupscompletedata=groupscompletedata
        self.groupsnames_=groupsnames_
        self.S=S
        self.groupnum=groupnum
        self.p=len(blocktypes)
        self.Datapreprocessing = Datapreprocessing


    # initialization
    def betainitial(self):
        #beta = np.ones((len(self.allfeatures), 1))
        beta=[]
        blocks_beta = dict()
        for blocktype in self.blocktypes:
            block_beta= [1] * len(self.blocks_features[blocktype])
            blocks_beta[blocktype] = block_beta
            beta+=block_beta
        return beta,blocks_beta

    def alphainitial(self):
        #alpha = np.ones((self.groupnum * self.S, 1))
        alpha=[]
        groups_alpha = dict()
        for groupname in self.groupsnames_:
            n_groupi=self.groupscompletedata[groupname].shape[0]
            #group_alpha=[1/n_groupi]*self.p
            group_alpha = [1]*self.p
            groups_alpha[groupname] =group_alpha
            alpha+=group_alpha
        return alpha,groups_alpha

    def alphabetaalternating_f(self,fit_intercept):
        beta,blocks_beta=self.betainitial()
        alpha,groups_alpha=self.alphainitial()

        niter=0
        betaf=0
        objl = math.inf
        objlnew = math.inf
        objl_chain=[]
        if len(self.groupsnames_)<=1:
            betaf = beta_f(self.TrainData, self.blocktypes, self.Yname, self.groupscompletedata,
                           self.groupsnames_, self.blocks_features, groups_alpha)
            beta, blocks_beta, objlnew = betaf.betaupdate_f(self.lamda, self.coefspenalty,
                                                            fit_intercept)
            self.betaf = betaf
            self.groups_alpha = groups_alpha
            self.alpha = alpha
            self.blocks_beta = blocks_beta
            self.beta = beta
            self.objl_chain = objlnew
            self.niter = niter
        else:
            while objlnew<=objl and niter<100:
                niter += 1
                objl=objlnew
                # store
                groups_alphaNew = groups_alpha
                alphaNew = alpha

                # updata beta given alpha
                betaf = beta_f(self.TrainData, self.blocktypes, self.Yname, self.groupscompletedata,
                               self.groupsnames_, self.blocks_features, groups_alpha)
                beta, blocks_beta, objlnew = betaf.betaupdate_f(self.lamda, self.coefspenalty,fit_intercept)

                # updata alpha given beta
                alphaf = alpha_f(self.TrainData, self.blocktypes, self.Yname, self.groupscompletedata,
                                 self.groupsnames_, self.blocks_features, blocks_beta)
                alpha,groups_alpha=alphaf.alphaupdate_f(self.mu,self.blockwisepenalty)


                objl_chain.append(objlnew)

            ###
            # updata beta given alpha
            betaNewf = beta_f(self.TrainData, self.blocktypes, self.Yname, self.groupscompletedata,
                           self.groupsnames_, self.blocks_features, groups_alphaNew)
            betaNew, blocks_betaNew, objlnew = betaNewf.betaupdate_f(self.lamda, self.coefspenalty, fit_intercept)


            self.betaf=betaNewf
            self.groups_alpha=groups_alphaNew
            self.alpha = alphaNew
            self.blocks_beta=blocks_betaNew
            self.beta = betaNew
            self.objl_chain=objl_chain
            self.niter=niter
        return self
    ############ predict for some group
    def predictgroup_f(self,onegroupcompleteTestdata,onegroup_alpha):
        alphax=self.betaf.onegroups_z_f(onegroupcompleteTestdata, onegroup_alpha)

        #predYgroup=np.dot(alphax,self.beta.reshape(len(self.beta),1))+self.betaf.intercept_

        alphax_dataframe= pd.DataFrame(alphax,columns=self.allfeatures)
        predYgroup=self.betaf.model.predict(alphax_dataframe).reshape(onegroupcompleteTestdata.shape[0],1)

        return predYgroup

    def predict_f(self,TestData):
        TestData = ScaleData_f(TestData, self.allfeatures, self.scaler)
        TestData.index=range(TestData.shape[0])

        groupsTestdata,groupcompleteTestdata,_=self.Datapreprocessing.groupsTestdata_f(TestData,self.groupsnames_)

        predY = np.empty((TestData.shape[0], 1))
        ntest=0
        for groupname_ in self.groupsnames_:
            if groupname_ not in groupcompleteTestdata.keys():
                continue
            onegroupcompleteTestdata=groupcompleteTestdata[groupname_]
            onegroup_alpha=self.groups_alpha[groupname_]
            indxTestData=onegroupcompleteTestdata.index
            predYgroup=self.predictgroup_f(onegroupcompleteTestdata,onegroup_alpha)
            predY[indxTestData]=predYgroup
            ntest+=onegroupcompleteTestdata.shape[0]
        predY=predY.reshape(1, len(predY))[0]
        if  len(predY)!=ntest:
            print('error in predY')
        predbinaryY=(predY>0).astype(int)

        return predY,predbinaryY


