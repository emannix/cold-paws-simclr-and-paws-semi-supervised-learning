# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# https://github.com/facebookresearch/suncet/blob/main/src/utils.py

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.types import STEP_OUTPUT

from pdb import set_trace as pb

class MyDDPStrategy(DDPStrategy):

    def __init__(self):
        super(MyDDPStrategy, self).__init__(find_unused_parameters=True)
        
    def predict_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.predict_step_context():
            # return self.model.module.module.predict_step(*args, **kwargs)
            try:
                return self.model.module._forward_module.predict_step(*args, **kwargs)
            except:
                return self.model.predict_step(*args, **kwargs)
