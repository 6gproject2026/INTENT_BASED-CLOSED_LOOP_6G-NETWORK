from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response

import subprocess
import json
import time
import re

latency_instance_name = 'latency_api_app'


class Network6GMonitor(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(Network6GMonitor, self).__init__(*args, **kwargs)

        wsgi = kwargs['wsgi']
        wsgi.register(LatencyController,
                      {latency_instance_name: self})

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):

        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()

        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]

        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions)]

        mod = parser.OFPFlowMod(datapath=datapath,
                                priority=0,
                                match=match,
                                instructions=inst)

        datapath.send_msg(mod)


class LatencyController(ControllerBase):

    def __init__(self, req, link, data, **config):
        super(LatencyController, self).__init__(req, link, data, **config)

    @route('latency', '/latency/{src}/{dst}', methods=['GET'])
    def get_latency(self, req, **kwargs):

        src = kwargs['src']
        dst = kwargs['dst']

        ip_map = {
            "URLLC": "20.0.0.1",
            "eMBB": "20.0.0.2",
            "mMTC": "20.0.0.3",
            "MNR_SVR" : "18.0.0.1"
        }

        latency = None
        loss = None

        try:

            dst_ip = ip_map[dst]

            pid_cmd = "pgrep -f 'mininet:%s'" % src
            pid = subprocess.check_output(pid_cmd, shell=True)
            pid = pid.strip().split('\n')[0]

            ping_cmd = "mnexec -a %s ping -c 4 %s" % (pid, dst_ip)
            out = subprocess.check_output(ping_cmd, shell=True)

            rtt = re.search(
                r'rtt min/avg/max/mdev = [\d\.]+/([\d\.]+)/', out)

            if rtt:
                latency = rtt.group(1)

            loss_match = re.search(
                r'(\d+)% packet loss', out)

            if loss_match:
                loss = loss_match.group(1)

        except:
            latency = None
            loss = None

        epoch_time = time.time()

        readable_time = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(epoch_time)
        )

        ms = int((epoch_time % 1) * 1000)

        timestamp = "%s.%03d" % (readable_time, ms)

        body = json.dumps({
            "src": src,
            "dst": dst,
            "latency_ms": latency,
            "packet_loss_percent": loss,
            "timestamp": timestamp,
            "epoch": epoch_time
        })

        return Response(content_type='application/json', body=body)
