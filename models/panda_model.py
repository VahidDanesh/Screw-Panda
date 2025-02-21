import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3
from typing import Union
import os
from config.config import PANDA_VIRTUAL


def create_virtual_panda(urdf_path: Union[str, None] = None) -> rtb.models.Panda:
    """
    Attaches a virtual finger link to a Panda robot model and returns the modified Panda.

    Args:
        urdf_path (str): The path to the URDF file containing the virtual finger.

    Returns:
        rtb.models.Panda: The modified Panda robot model.
    """
    if urdf_path is None:
        urdf_path = os.path.join(os.getcwd(), PANDA_VIRTUAL)
    
    panda = rtb.models.Panda()
    panda_virtual = rtb.Robot.URDF(urdf_path, 'panda_hand')

    panda_virtual.addconfiguration('qr', np.append(panda.qr, 0))
    panda_virtual.addconfiguration('qz', np.append(panda.qz, 0))
    panda_virtual.qr = np.append(panda.qr, 0)
    panda_virtual.qz = np.append(panda.qz, 0)

    panda_virtual.q = panda_virtual.qr

    #rebuild the ETS.
    panda_virtual.ets()

    return panda_virtual
