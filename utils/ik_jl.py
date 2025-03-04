import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import IKSolver, IKSolution
from roboticstoolbox.tools import *
import spatialmath as sm
from spatialmath import SE3
from typing import Union, Tuple

class IK_JL(IKSolver):
    """
    Joint Limit Avoidance Numerical Inverse Kinematics Solver

    A class which provides functionality to perform numerical inverse kinematics (IK)
    using the joint limit avoidance method described in 
    the Robotics: Modelling, Planning and Control by Siciliano, Bruno and Sciavicco, Lorenzo.

    Parameters
    ----------
    name
        The name of the IK algorithm
    ilimit
        How many iterations are allowed within a search before a new search
        is started
    slimit
        How many searches are allowed before being deemed unsuccessful
    tol
        Maximum allowed residual error E
    mask
        A 6 vector which assigns weights to Cartesian degrees-of-freedom
        error priority
    joint_limits
        Reject solutions with joint limit violations
    seed
        A seed for the private RNG used to generate random joint coordinate
        vectors
    K
        The gain for the main IK algorithm
    k0
        The gain for joint limit avoidance
        
    Examples
    --------
    .. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> panda = rtb.models.Panda().ets()
    >>> solver = rtb.IK_GN(pinv=True)
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> solver.solve(panda, Tep)
    
    Notes
    -----
    The algorithm is a numerical inverse kinematics solver which uses the Jacobian
    to iteratively reduce the error between the current end-effector pose and the
    desired end-effector pose. The algorithm uses a joint limit avoidance term to
    prevent joint limits from being violated.
    
    The algorithm is based on the following update rule:
    
    .. math::
        q_{dot} = J^{+}e + (I - J^{+}J)q_{0}
        
    where :math:`q_{dot}` is the joint velocities, :math:`J^{+}` is the pseudo-inverse
    of the Jacobian, :math:`e` is the error between the current and desired end-effector
    poses, :math:`q_{0}` is the joint limit avoidance term, and :math:`J` is the Jacobian.
    
    The joint limit avoidance term is calculated as follows:
    
    .. math::
        q_{0} = k_{0} \cdot dw_{dq}
        
    where :math:`k_{0}` is the gain for the joint limit avoidance term and :math:`dw_{dq}`
    is the gradient of the joint limit avoidance function.
    
    The gradient of the joint limit avoidance function is calculated as follows:
    
    .. math::
        dw_{dq} = -\frac{q - q_{mid}}{(\frac{q_{range}}{2})^{2}}
        
    where :math:`q` is the current joint coordinate vector, :math:`q_{mid}` is the midpoint
    of the joint limits, and :math:`q_{range}` is the range of the joint limits.
    
    References
    ----------
    - Robotics: Modelling, Planning and Control by Siciliano, Bruno and Sciavicco, Lorenzo
    


    """

    def __init__(
        self,
        name: str = "IK Solver",
        ilimit: int = 30,
        slimit: int = 100,
        tol: float = 1e-6,
        mask: Union[ArrayLike, None] = None,
        joint_limits: bool = True,
        seed: Union[int, None] = None,
        K: Union[ArrayLike, float, None] = 1.0,
        k0: Union[ArrayLike, float, None] = 0.1,
        𝒱ep: ArrayLike = np.empty(6),
        **kwargs,
    ):
        super().__init__(
            name=name,
            ilimit=ilimit,
            slimit=slimit,
            tol=tol,
            mask=mask,
            joint_limits=joint_limits,
            seed=seed,
            **kwargs,
        )

        self.K = K
        self.k0 = k0
        self.Vep = 𝒱ep
        self.name = f"Joint Limit Avoidance"

    def step(
        self, ets: "rtb.ETS", Tep: np.ndarray, q: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        r"""
        Performs a single iteration of the Joint Limit Avoidance IK method

        Parameters
        ----------
        ets
            The ETS representing the manipulators kinematics
        Tep
            The desired end-effector pose
        q
            The current joint coordinate vector

        Returns
        -------
        E
            The new error value
        q
            The new joint coordinate vector
        """

        Te = ets.eval(q)
        e, E = self.error(Te, Tep)

        J = ets.jacob0(q)
        J_pinv = np.linalg.pinv(J)

        # Calculate joint limit avoidance term
        dw_dq = self._calculate_dw_dq(ets, q)
        
        q0 = self.k0 * dw_dq

        # Calculate joint velocities
        

        q_dot = J_pinv @ (self.Vep + self.K * e) + (np.eye(ets.n) - J_pinv @ J) @ q0

        # Update joint positions
        
        q[ets.jindices] += q_dot

        return E, q[ets.jindices]

    def _calculate_dw_dq(self, ets: "rtb.ETS", q: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the joint limit avoidance function

        Parameters
        ----------
        ets
            The ETS representing the manipulators kinematics
        q
            The current joint coordinate vector

        Returns
        -------
        dw_dq
            The gradient of the joint limit avoidance function
        """
        n = ets.n
        qlim = ets.qlim
        
        q_mid = (qlim[1] + qlim[0]) / 2
        q_range = qlim[1] - qlim[0]
        
        # Vectorized calculation with singularity protection
        with np.errstate(divide='ignore', invalid='ignore'):
            dw_dq = -1/n * (q - q_mid) / (q_range / 2)**2
            
        # Handle unlimited joints (range = 0)
        dw_dq[np.isinf(dw_dq)] = 0  
        dw_dq[np.isnan(dw_dq)] = 0
        
        return dw_dq