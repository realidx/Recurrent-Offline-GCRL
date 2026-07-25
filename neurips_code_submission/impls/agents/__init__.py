from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.hiql import HIQLAgent
from agents.sac import SACAgent
from agents.saw import SAWAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    hiql=HIQLAgent,
    sac=SACAgent,
    saw=SAWAgent,
)
