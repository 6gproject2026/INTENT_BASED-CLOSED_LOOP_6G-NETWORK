import time


def _run_bg(host, command):
    host.cmd("%s > /tmp/%s_iperf.log 2>&1 &" % (command, host.name))


def _kill_iperf(host):
    host.cmd("pkill -f iperf")


def start_traffic(
    net,
    duration_s=1800,
    urllc_bw="10M",
    embb_bw="10M",
    mmtc_bw="10M",
):
    """Start UDP traffic flows for the full training window."""
    servers = {
        "URLLC": 5001,
        "eMBB": 5002,
        "mMTC": 5003,
    }
    clients = {
        "G6_D1": ("20.0.0.1", 5001, urllc_bw),
        "G6_D2": ("20.0.0.2", 5002, embb_bw),
        "G6_IOT_D": ("20.0.0.3", 5003, mmtc_bw),
    }

    for name in list(servers.keys()) + list(clients.keys()):
        _kill_iperf(net.get(name))

    for name, port in servers.items():
        host = net.get(name)
        _run_bg(host, "iperf -s -u -p %s -i 1" % port)

    time.sleep(1)

    for name, (dst_ip, port, bw) in clients.items():
        host = net.get(name)
        _run_bg(host, "iperf -c %s -u -p %s -b %s -t %s" % (dst_ip, port, bw, duration_s))

    print(
        "Traffic started: duration=%ss urllc=%s embb=%s mmtc=%s"
        % (duration_s, urllc_bw, embb_bw, mmtc_bw)
    )


def stop_traffic(net):
    for name in ["URLLC", "eMBB", "mMTC", "G6_D1", "G6_D2", "G6_IOT_D"]:
        _kill_iperf(net.get(name))
    print("Traffic stopped")


if "net" in globals():
    start_traffic(net)
else:
    print("Run this from Mininet CLI: py exec(open('traffic_runner.py').read())")
