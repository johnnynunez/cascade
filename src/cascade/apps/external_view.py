"""Read-only attachment to an independently supervised tailnet camera service."""
from __future__ import annotations
import ipaddress
import json
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener


class ExternalLiveViewController:
    server = None
    mode = 'supervised'
    idle_timeout_s = None

    def __init__(self, url):
        parsed = urlsplit(url)
        if (parsed.scheme!='http' or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ('','/') or not parsed.hostname
                or ipaddress.ip_address(parsed.hostname) not in ipaddress.ip_network('100.64.0.0/10')):
            raise ValueError('External camera service must be an exact Tailscale HTTP origin')
        self.url=url.rstrip('/')+'/'

    def status(self):
        try:
            with build_opener(ProxyHandler({})).open(self.url+'state',timeout=2) as response:
                state=json.loads(response.read(1024*1024))
            cameras=state.get('cameras',{})
            if isinstance(cameras,list):cameras={c['name']:c for c in cameras}
            online=all(cameras.get(n,{}).get('online') is True for n in ('kitchen','side','worktop'))
            return {'ok':online,'open':online,'url':self.url,'mode':self.mode,
                    'read_only':True,'supervised':True,'cameras':list(cameras)}
        except (OSError,ValueError,KeyError):
            return {'ok':False,'open':False,'url':self.url,'mode':self.mode,
                    'error':'The supervised camera service is not currently healthy'}

    def open(self,reason=''):
        return self.status()

    def close(self,reason=''):
        return {**self.status(),'note':'The independent camera service remains supervised; close its browser tab to hide it.'}

    @property
    def is_open(self):
        return self.status().get('open',False)

    def note_poll(self):pass
    def set_task_fn(self,fn):pass
    def set_cancel_fn(self,fn):pass
