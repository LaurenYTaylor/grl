from .bc import BCAgent
from .calql import CalQLAgent
from .cql import CQLAgent
from .iql import IQLAgent
from .sac import SACAgent
from .jsrl import JSRLAgent
from .jsrl_random import JSRLRandomAgent
from .pex import PEXAgent
from .jsrl_sac import JSRLSACAgent
from .grl_sac import GRLSACAgent

agents = {
    "bc": BCAgent,
    "iql": IQLAgent,
    "cql": CQLAgent,
    "calql": CalQLAgent,
    "sac": SACAgent,
    "jsrl": JSRLAgent,
    "jsrl_random": JSRLRandomAgent,
    "jsrl_sac": JSRLSACAgent,
    "pex": PEXAgent,
    "grl": GRLSACAgent,
}
