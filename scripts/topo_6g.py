"""6G closed-loop test fabric.

Reconstructed from prod.json (environment.startup_setup.routing addresses) and
the out_port values in a known-good models/next_hops.json. LINK ORDER MATTERS:
port numbers are assigned in creation order, and the captured next-hop map
depends on them:

    core-eth1 -> agg (12.0.0.0/24)   core-eth2 -> sp1 (13.0.0.0/24)
    core-eth3 -> sp2 (14.0.0.0/24)
    RAN-eth1..4 -> hosts, RAN-eth5 -> agg
    lf1-eth1 -> sp1, lf1-eth2 -> sp2, lf1-eth3..6 -> hosts

Fabric link capacities match prod.json: main path (core-sp1-lf1) 20 Mbps,
backup path (core-sp2-lf1) 10 Mbps. --link tc is required for these to apply.

Run (inside the Mininet/Ryu container, BEFORE starting ryu-manager):

    mn -c
    mn --custom /root/topo_6g.py --topo sixg --link tc \
       --controller=remote,ip=127.0.0.1,port=6633 \
       --switch=ovs,protocols=OpenFlow13

NOTE: Topo sorts links, so creation order here does not fully determine port
numbers -- lf1's uplinks land on eth5/eth6, not eth1/eth2. core-eth1/2/3 ->
agg/sp1/sp2 does hold, which is what the failover next-hop depends on.
"""

from mininet.topo import Topo

MAIN_BW = 20
BACKUP_BW = 10


class SixGTopo(Topo):
    def build(self):
        RAN = self.addSwitch("RAN", dpid="0000000000000010")
        agg = self.addSwitch("agg", dpid="0000000000000020")
        core = self.addSwitch("core", dpid="0000000000000030")
        sp1 = self.addSwitch("sp1", dpid="0000000000000040")
        sp2 = self.addSwitch("sp2", dpid="0000000000000041")
        lf1 = self.addSwitch("lf1", dpid="0000000000000050")

        # RAN access ports 1-4
        self.addLink(self.addHost("G6_D1", ip="10.0.0.1/24"), RAN)
        self.addLink(self.addHost("G6_D2", ip="10.0.0.2/24"), RAN)
        self.addLink(self.addHost("G6_IOT_D", ip="10.0.0.3/24"), RAN)
        self.addLink(self.addHost("MNR_D", ip="17.0.0.1/24"), RAN)

        # Fabric uplinks, in the order that fixes the port numbering above
        self.addLink(RAN, agg)                    # 11.0.0.0/24
        self.addLink(agg, core)                   # 12.0.0.0/24
        self.addLink(core, sp1, bw=MAIN_BW)       # 13.0.0.0/24  main
        self.addLink(core, sp2, bw=BACKUP_BW)     # 14.0.0.0/24  backup
        self.addLink(sp1, lf1, bw=MAIN_BW)        # 15.0.0.0/24  main
        self.addLink(sp2, lf1, bw=BACKUP_BW)      # 16.0.0.0/24  backup

        # lf1 access ports 3-6
        self.addLink(self.addHost("URLLC", ip="20.0.0.1/24"), lf1)
        self.addLink(self.addHost("eMBB", ip="20.0.0.2/24"), lf1)
        self.addLink(self.addHost("mMTC", ip="20.0.0.3/24"), lf1)
        self.addLink(self.addHost("MNR_SVR", ip="18.0.0.1/24"), lf1)


topos = {"sixg": (lambda: SixGTopo())}
