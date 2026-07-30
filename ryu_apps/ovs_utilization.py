# -*- coding: utf-8 -*-

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from ryu.topology import event
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response
import time
import json

LINK_API_NAME = 'link_util_api'


class OVSLinkUtilization(app_manager.RyuApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):

        super(OVSLinkUtilization, self).__init__(*args, **kwargs)

        wsgi = kwargs['wsgi']
        wsgi.register(LinkUtilizationController, {LINK_API_NAME: self})

        self.datapaths = {}
        self.port_stats = {}
        self.port_speed = {}
        self.links = []

        # MAP DPID → SWITCH NAME
        self.name_map = {
            16: "RAN",
            32: "agg",
            48: "core",
            64: "sp1",
            65: "sp2",
            80: "lf1"
        }

        # ORDERING OF SWITCHES
        self.order = {
            "RAN": 1,
            "agg": 2,
            "core": 3,
            "sp1": 4,
            "sp2": 5,
            "lf1": 6
        }

        self.monitor_thread = hub.spawn(self._monitor)

    # -------------------------
    # Register switches
    # -------------------------

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])

    def _state_change_handler(self, ev):

        datapath = ev.datapath

        if ev.state == MAIN_DISPATCHER:
            self.datapaths[datapath.id] = datapath

        elif ev.state == DEAD_DISPATCHER:

            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]

    # -------------------------
    # Topology discovery
    # -------------------------

    @set_ev_cls(event.EventLinkAdd)

    def get_link(self, ev):

        src = ev.link.src
        dst = ev.link.dst

        self.links.append((src.dpid, src.port_no, dst.dpid, dst.port_no))

    # -------------------------
    # Monitoring loop
    # -------------------------

    def _monitor(self):

        while True:

            for dp in self.datapaths.values():
                self._request_stats(dp)

            hub.sleep(1)

    def _request_stats(self, datapath):

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    # -------------------------
    # Receive port stats
    # -------------------------

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)

    def _port_stats_reply_handler(self, ev):

        dpid = ev.msg.datapath.id

        for stat in ev.msg.body:

            port = stat.port_no
            key = (dpid, port)

            tx = stat.tx_bytes
            rx = stat.rx_bytes
            now = time.time()

            if key in self.port_stats:

                prev_tx, prev_rx, prev_time = self.port_stats[key]
                interval = now - prev_time

                if interval > 0:

                    tx_speed = (tx - prev_tx) * 8 / interval / 1000000
                    rx_speed = (rx - prev_rx) * 8 / interval / 1000000

                    self.port_speed[key] = (tx_speed, rx_speed)

            self.port_stats[key] = (tx, rx, now)


class LinkUtilizationController(ControllerBase):

    def __init__(self, req, link, data, **config):

        super(LinkUtilizationController, self).__init__(req, link, data, **config)

        self.app = data[LINK_API_NAME]

    @route('links', '/links/utilization', methods=['GET'])

    def list_links(self, req, **kwargs):

        result = []

        for (src_dpid, src_port, dst_dpid, dst_port) in self.app.links:

            src_key = (src_dpid, src_port)
            dst_key = (dst_dpid, dst_port)

            if src_key in self.app.port_speed and dst_key in self.app.port_speed:

                src_tx, _ = self.app.port_speed[src_key]
                _, dst_rx = self.app.port_speed[dst_key]

                traffic = min(src_tx, dst_rx)

                src_name = self.app.name_map.get(src_dpid, str(src_dpid))
                dst_name = self.app.name_map.get(dst_dpid, str(dst_dpid))

                result.append({
                    "link": "%s -> %s" % (src_name, dst_name),
                    "tx_mbps": round(traffic, 2)
                })

        # ORDER OUTPUT
        result.sort(key=lambda x: (
            self.app.order.get(x["link"].split(" -> ")[0], 100),
            self.app.order.get(x["link"].split(" -> ")[1], 100)
        ))

        body = json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "links": result
        })

        return Response(content_type='application/json', body=body)
