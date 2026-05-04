import threading
import time


def _run_bg(host, command):
    host.cmd("%s > /tmp/%s_iperf.log 2>&1 &" % (command, host.name))


def _kill_iperf(host):
    host.cmd("pkill -f iperf")


def _get_kill_fn():
    kill_fn = globals().get("_kill_iperf")
    if callable(kill_fn):
        return kill_fn

    def _kill(host):
        host.cmd("pkill -f iperf")

    return _kill


def start_traffic(
    net,
    duration_s=1800,
    urllc_bw="10M",
    embb_bw="10M",
    mmtc_bw="10M",
):
    """Start UDP traffic flows for the full training window."""
    def run_bg(host, command):
        host.cmd("%s > /tmp/%s_iperf.log 2>&1 &" % (command, host.name))

    def kill(host):
        host.cmd("pkill -f iperf")

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
        kill(net.get(name))

    for name, port in servers.items():
        host = net.get(name)
        run_bg(host, "iperf -s -u -p %s -i 1" % port)

    time.sleep(1)

    for name, (dst_ip, port, bw) in clients.items():
        host = net.get(name)
        run_bg(host, "iperf -c %s -u -p %s -b %s -t %s" % (dst_ip, port, bw, duration_s))

    print(
        "Traffic started: duration=%ss urllc=%s embb=%s mmtc=%s"
        % (duration_s, urllc_bw, embb_bw, mmtc_bw)
    )


def stop_traffic(net):
    def kill(host):
        host.cmd("pkill -f iperf")

    for name in ["URLLC", "eMBB", "mMTC", "G6_D1", "G6_D2", "G6_IOT_D"]:
        kill(net.get(name))
    print("Traffic stopped")


def traffic_status(net, tail_lines=3):
    hosts = ["URLLC", "eMBB", "mMTC", "G6_D1", "G6_D2", "G6_IOT_D"]
    for name in hosts:
        host = net.get(name)
        procs = host.cmd("pgrep -fa iperf")
        procs = procs.strip() if procs else ""
        print("%s iperf: %s" % (name, procs if procs else "not running"))
        log_path = "/tmp/%s_iperf.log" % host.name
        tail = host.cmd("tail -n %s %s 2>/dev/null" % (int(tail_lines), log_path))
        tail = tail.strip() if tail else ""
        if tail:
            print("%s log tail:\n%s" % (name, tail))


DEFAULT_PLAN = [
    {"duration_s": 900, "urllc_bw": "30M", "embb_bw": "10M", "mmtc_bw": "10M"},
    {"duration_s": 900, "urllc_bw": "15M", "embb_bw": "20M", "mmtc_bw": "5M"},
    {"duration_s": 900, "urllc_bw": "5M", "embb_bw": "30M", "mmtc_bw": "20M"},
]

_traffic_plan_thread = None
_traffic_plan_stop = None


def _run_plan(net, plan, loop, stop_event):
    while not stop_event.is_set():
        for entry in plan:
            if stop_event.is_set():
                break
            duration_s = int(entry.get("duration_s", 900))
            urllc_bw = entry.get("urllc_bw", "10M")
            embb_bw = entry.get("embb_bw", "10M")
            mmtc_bw = entry.get("mmtc_bw", "10M")

            start_traffic(
                net,
                duration_s=duration_s,
                urllc_bw=urllc_bw,
                embb_bw=embb_bw,
                mmtc_bw=mmtc_bw,
            )

            remaining = duration_s
            while remaining > 0 and not stop_event.is_set():
                sleep_for = 5 if remaining > 5 else remaining
                time.sleep(sleep_for)
                remaining -= sleep_for

            stop_traffic(net)

        if not loop:
            break


def start_traffic_plan(net, plan=None, loop=True):
    """Cycle through multiple traffic profiles in the background."""
    global _traffic_plan_thread, _traffic_plan_stop
    if _traffic_plan_thread is not None and _traffic_plan_thread.is_alive():
        print("Traffic plan already running")
        return

    try:
        default_plan = DEFAULT_PLAN
    except NameError:
        default_plan = [
            {"duration_s": 900, "urllc_bw": "30M", "embb_bw": "10M", "mmtc_bw": "10M"},
            {"duration_s": 900, "urllc_bw": "15M", "embb_bw": "20M", "mmtc_bw": "5M"},
            {"duration_s": 900, "urllc_bw": "5M", "embb_bw": "30M", "mmtc_bw": "20M"},
        ]

    plan = plan if plan is not None else default_plan
    stop_event = threading.Event()
    thread = threading.Thread(target=_run_plan, args=(net, plan, loop, stop_event))
    thread.daemon = True
    thread.start()

    _traffic_plan_stop = stop_event
    _traffic_plan_thread = thread
    print("Traffic plan started")


def stop_traffic_plan(net):
    """Stop the active traffic plan and kill iperf processes."""
    global _traffic_plan_thread, _traffic_plan_stop
    if _traffic_plan_stop is not None:
        _traffic_plan_stop.set()
    stop_traffic(net)
    _traffic_plan_thread = None
    _traffic_plan_stop = None


def _expose_to_main():
    try:
        import __main__

        __main__.start_traffic = start_traffic
        __main__.stop_traffic = stop_traffic
        __main__.traffic_status = traffic_status
        __main__.start_traffic_plan = start_traffic_plan
        __main__.stop_traffic_plan = stop_traffic_plan
    except Exception:
        pass


_expose_to_main()


print("traffic_runner loaded. Use start_traffic(net, ...) or start_traffic_plan(net, ...)")
