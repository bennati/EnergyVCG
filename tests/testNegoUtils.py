import unittest
from nego.src.utilsnego import is_mediator_biased,lo_hi,split_bids
import functools
import numpy as np

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
