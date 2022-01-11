

from PIML.util.basebox import BaseBox
from testbase import TestBase
from PIML.util.basebox import BaseBox

class Test_BaseBox(TestBase):

    def test_init(self):
        BB = BaseBox()

    def test_init_bnds(self):
        BB = BaseBox()
        BB.init_bnds()
        self.assertIsNotNone(BB.DPhyRng)
