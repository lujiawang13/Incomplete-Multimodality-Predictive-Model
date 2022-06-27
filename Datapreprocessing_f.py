import pandas as pd
import numpy as np
import sys

from fnmatch import fnmatch
from itertools import combinations
from copy import copy

def diff(first, second):
    return [item for item in first if item not in second]

def blocksfeatures_f(TrainData, blocktypes):
    colnames = list(TrainData.columns.values)
    allfeatures = []
    blocks_features = dict()
    # blocks_features = {}
    for blocktype in blocktypes:
        subnames = blocktype.split('*')
        block_features = []
        for subname in subnames:
            subblock_features = [feature for feature in colnames if fnmatch(feature, "*" + subname + "*")]
            block_features += subblock_features
        # blocks_features[blocktype] =block_features
        blocks_features.update({blocktype: block_features})
        allfeatures += block_features
    return allfeatures,blocks_features

class Datapreprocessing_f():
    # init method or constructor
    def __init__(self, TrainData, blocktypes,Yname):
        self.TrainData = TrainData
        self.blocktypes=blocktypes
        self.Yname=Yname

        self.allfeatures, self.blocks_features=blocksfeatures_f(TrainData, blocktypes)
        # self.allfeatures=allfeatures
        # self.blocks_features=blocks_features

    def Blockcombinations_f(self):
        Blockcombinations = []
        for r in range(1, len(self.blocktypes) + 1):
            combinations_object = combinations(self.blocktypes, r)
            combinations_list = list(combinations_object)
            Blockcombinations += combinations_list
        return Blockcombinations
    #################################

    def onegroupcompletedata_f(self,onegroupdata,groupname):
        onen=onegroupdata.shape[0]
        blocktypes = self.blocktypes
        colnames = list(self.TrainData.columns.values)
        othernames=diff(colnames,self.allfeatures)
        onegroupcompletedata=pd.DataFrame()
        for blocktype in blocktypes:
            block_features = self.blocks_features[blocktype]
            onep=len(block_features)
            if blocktype in groupname:
                oneTrainData = onegroupdata.loc[:, block_features]
            else:
                oneTrainData=pd.DataFrame(np.zeros(onen*onep).reshape(onen,onep), columns=block_features)

            onegroupcompletedata = pd.concat([onegroupcompletedata, oneTrainData.reset_index(drop=True)], axis=1)

        onegroupcompletedata = pd.concat([onegroupcompletedata, onegroupdata.loc[:,othernames].reset_index(drop=True)], axis=1)

        return onegroupcompletedata

    def groupsearch_f(self,TrainData,groupname):
        blocktypes=self.blocktypes
        blocktypescopy = copy(blocktypes)
        # existing
        for onename in groupname:
            blocktypescopy.remove(onename)
            onefeature = self.blocks_features[onename][1]
            TrainData = TrainData.loc[np.isnan(TrainData.loc[:, onefeature]) == False, :]
        # missing
        for onename in blocktypescopy:
            onefeature = self.blocks_features[onename][1]
            TrainData = TrainData.loc[np.isnan(TrainData.loc[:, onefeature]), :]
        onegroupdata=TrainData
        # complete features; add zero matrix
        onegroupcompletedata=self.onegroupcompletedata_f(onegroupdata,groupname)
        onegroupcompletedata.index=onegroupdata.index
        return onegroupdata,onegroupcompletedata



    def groupsdata_f(self):
        Blockcombinations=self.Blockcombinations_f()
        #self.blocksfeatures_f()

        n = 0
        groupsdata = dict()
        groupcompletedata= dict()
        groupnames_=[]
        for groupname in Blockcombinations:
            groupname_ = '_'.join(groupname)

            onegroupdata,onegroupcompletedata = self.groupsearch_f(self.TrainData,groupname)

            if onegroupdata.shape[0]==0:
                continue
            groupsdata[groupname_] = onegroupdata
            groupcompletedata[groupname_] = onegroupcompletedata
            groupnames_.append(groupname_)
            n = n + onegroupdata.shape[0]

        if self.TrainData.shape[0] != n:
            print('error in sample size in groups')
        if len(groupsdata)<1:
            sys.exit('Warning: make sure block numbers >1 & having missing blocks ')
        #self.groupnames_=groupnames_
        return groupsdata,groupcompletedata,groupnames_

    # Test data
    def groupsTestdata_f(self,TestData,groupsnames_):

        n = 0
        Testgroupsnames_=[]
        groupsdata = dict()
        groupcompletedata= dict()
        for groupname_ in groupsnames_:
            groupname = groupname_.split('_')

            onegroupdata,onegroupcompletedata = self.groupsearch_f(TestData,groupname)
            if onegroupdata.shape[0]==0:
                continue

            groupsdata[groupname_] = onegroupdata
            groupcompletedata[groupname_] = onegroupcompletedata
            Testgroupsnames_.append(groupname_)
            n = n + onegroupdata.shape[0]

        if TestData.shape[0] != n:
            print('error in sample size in test groups')

        return groupsdata,groupcompletedata,Testgroupsnames_
