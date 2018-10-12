import unittest
from nego.src.utilsnego import is_mediator_biased,lo_hi,split_bids,read_income_data,compute_incomes
import functools
import numpy as np
import pandas as pd

class TestNegoUtils(unittest.TestCase):
    """
    This class testes the measurement generation
    """

    def __init__(self, *args, **kwargs):
        super(TestNegoUtils, self).__init__(*args, **kwargs)

    def test_mediator_not_biased(self):
        assert(is_mediator_biased(0.0) is None)

    def test_mediator_biased(self):
        r=10000;d=0.7;tol=0.01
        s=sum([is_mediator_biased(d) for _ in range(r)])/r
        print(s)
        assert(abs(s-d)<tol)

    def test_lo_hi(self):
        r=10000;tol=0.01
        lo=np.random.uniform();hi=np.random.uniform()
        s=sum([lo_hi(True,lo,hi) for _ in range(r)])/r
        assert(abs(s-lo)<tol)
        s=sum([lo_hi(False,lo,hi) for _ in range(r)])/r
        assert(abs(s-hi)<tol)

    def test_split_bids(self):
        bid={"asd":"asd","value":np.random.uniform(10)}
        ans=split_bids([bid],splitsize=1.0)
        assert(all([i["asd"]=="asd" for i in ans])) # is a duplicated
        assert(len(ans)==np.floor(bid['value'])+1)  # split a correct number of times

    def test_income_generation(self):
        N=10000
        byincome,byvillage=read_income_data("./datasets")
        castes=[np.random.uniform()<byvillage['Dalit_prop'].mean() for _ in range(N)] # determine the caste, true means low caste
        incomes=compute_incomes(byincome,castes) # compute incomes based on real data
        df=pd.DataFrame(data={'income':incomes,'bin':np.digitize(incomes,bins=byincome['income_min'].unique())}) # label them according to the corresponding income interval
        df=df.groupby('bin').mean().reset_index() # compute mean over each income interval
        df['bin']=df['bin']-1
        income_means=pd.DataFrame(np.unique((byincome['income_min']+byincome['income_max'])/2),columns=["income_mean"]).reset_index()
        df=pd.merge(df,income_means,left_on='bin',right_on='index')
        assert(all(abs(df['income']-df['income_mean'])<df['income_mean']*0.1)) # make sure the generated incomes do not deviate too much from the actual means
