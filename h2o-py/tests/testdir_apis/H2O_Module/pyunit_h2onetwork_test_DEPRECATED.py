from __future__ import print_function
import sys
sys.path.insert(1,"../../../")
from tests import pyunit_utils
import h2o

def h2onetwork_test():
    """
    Python API test: h2o.network_test()
    Deprecated, use h2o.cluster().network_test().
    """
    h2o.network_test()    # no return type

if __name__ == "__main__":
    pyunit_utils.standalone_test(h2onetwork_test)
else:
    h2onetwork_test()
