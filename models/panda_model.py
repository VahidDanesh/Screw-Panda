import roboticstoolbox as rtb
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

    if panda_virtual._getlink('panda_finger_virtual') is not None:
        virtual_link = panda_virtual.link_dict['panda_finger_virtual']
        panda.link_dict['panda_finger_virtual'] = virtual_link
        panda.grippers[0].links.append(virtual_link)
        panda.grippers[0].links[0].children.append(virtual_link)
    else:
        print("Warning: 'panda_finger_virtual' link not found in the URDF.")



    #rebuild the ETS.
    panda.ets()

    return panda
