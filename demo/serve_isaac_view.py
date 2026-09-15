"""Read-only kitchen view; one owned transport automatically reconnects.

No controller, model, simulator launcher, physics exec or actuator is used.
"""
from __future__ import annotations
import json
import math
import os
import signal
import threading
import time
import cv2
import numpy as np
from cascade.apps.stream_server import StreamServer
from cascade.sim.bridge_client import BridgeClient

CAMERAS = (("kitchen", "proof"), ("worktop", "cam0"), ("side", "side"))
URL = os.environ.get("CASCADE_EXTERNAL_VIEW_URL", "http://127.0.0.1:8091/")


class ViewStream:
    has_depth = True
    def __init__(self, rig, name): self._rig, self.name = rig, name
    def latest(self): return self._rig.latest(self.name)
    @property
    def fps(self): return self._rig.stats()[self.name]["fps"]
    @property
    def last_error(self): return self._rig.stats()[self.name]["error"]
    def overlay(self):
        return [], "OpenClaw Demo - read only" if self.latest() is not None else "Offline - reconnecting"


class RecoveringViewRig:
    """Stable HTTP-facing slots, rebuilt BridgeClient, one joinable capture thread."""
    def __init__(self, *, client_factory=None, stop_event=None, clock=time.monotonic,
                 rate_hz=8, stale_s=2, initial_backoff_s=.25, max_backoff_s=5):
        if not 0 < initial_backoff_s <= max_backoff_s <= 5 or not 0 < stale_s <= 5:
            raise ValueError("Recovery/staleness bounds must be positive and <=5 seconds")
        self._factory = client_factory or (lambda: BridgeClient(host="127.0.0.1",port=8611,timeout_s=1))
        self._stop = stop_event or threading.Event()
        self._clock, self._period, self._stale = clock, 1/max(1,rate_hz), stale_s
        self._initial, self._maximum = initial_backoff_s, max_backoff_s
        self._lock = threading.Lock()
        self._thread = None
        self._online = False
        self._error = "Connecting to Isaac Sim"
        self._retry_at = None
        self._attempts = self._generation = 0
        self._frames = {}
        self._seq = {n:0 for n,_ in CAMERAS}
        self._fps = {n:0.0 for n,_ in CAMERAS}
        self.streams = {n:ViewStream(self,n) for n,_ in CAMERAS}
    @property
    def names(self): return list(self.streams)
    @property
    def primary(self): return self.streams["kitchen"]
    def __iter__(self): return iter(self.streams.values())
    def __len__(self): return len(self.streams)
    def get(self,name): return self.primary if name is None else self.streams[name]
    def _fresh(self,frame):
        t=(frame.capture or {}).get("t") if frame is not None else None
        return isinstance(t,(float,int)) and math.isfinite(t) and -.25 <= self._clock()-t <= self._stale
    def latest(self,name):
        with self._lock:
            f=self._frames.get(name)
            return f if self._online and self._fresh(f) else None
    def stats(self):
        with self._lock:
            result={}
            for name in self.names:
                live=self._online and self._fresh(self._frames.get(name))
                result[name]={"fps":round(self._fps[name],1) if live else 0.0,
                    "frame_id":self._seq[name] if live else 0,"depth":True,"online":bool(live),
                    "error":None if live else self._error or "Stale capture"}
            return result
    def state(self):
        with self._lock:
            live={n:self._online and self._fresh(self._frames.get(n)) for n in self.names}
            online=all(live.values())
            return {"agent_status":"OpenClaw Demo - live" if online else "OpenClaw Demo - offline; reconnecting",
                "read_only":True,"stream_status":"online" if online else "offline",
                "capture":{n:self._frames[n].capture if live[n] else None for n in self.names},
                "bridge_error":None if online else self._error,
                "connection_attempts":self._attempts,"connection_generation":self._generation,
                "retry_in_s":max(0.0,self._retry_at-self._clock()) if self._retry_at else 0.0}
    def _offline(self,error,delay=None):
        with self._lock:
            self._online=False; self._error=str(error)[:240]
            self._frames.clear()
            self._retry_at=self._clock()+delay if delay is not None else None
    def _validate(self,frame,camera):
        if not self._fresh(frame): raise RuntimeError(f"{camera}: capture missing or stale")
        cap=frame.capture or {}; prop=cap.get("proprioception") or {}
        if (cap.get("camera")!=camera or prop.get("t")!=cap.get("t") or
            prop.get("time_source")!="physics_loop_monotonic" or not prop.get("robot_id")):
            raise RuntimeError(f"{camera}: capture identity or clock could not be verified")
    def _run(self):
        delay=self._initial
        try:
            while not self._stop.is_set():
                client=None
                try:
                    with self._lock: self._attempts+=1
                    client=self._factory(); client.connect()
                    if not client.ping(): raise RuntimeError("bridge ping failed")
                    previous={}
                    with self._lock: self._generation+=1
                    while not self._stop.is_set():
                        began=self._clock(); frames={}
                        for name,camera in CAMERAS:
                            if self._stop.is_set(): break
                            frame=client.observation(camera); self._validate(frame,camera)
                            if camera in previous and frame.capture["t"]<previous[camera]:
                                raise RuntimeError(f"{camera}: capture clock moved backwards")
                            frames[name]=frame
                        if self._stop.is_set(): break
                        now=self._clock()
                        with self._lock:
                            for name,camera in CAMERAS:
                                frame=frames[name]
                                if frame.capture["t"]!=previous.get(camera):
                                    self._seq[name]+=1
                                    old_stamp=previous.get(camera)
                                    measured=min(1/self._period,1/(frame.capture["t"]-old_stamp)) if old_stamp is not None else 0.0
                                    self._fps[name]=.9*self._fps[name]+.1*measured if old_stamp is not None else 0.0
                                    frame.frame_id=self._seq[name]; self._frames[name]=frame
                                previous[camera]=frame.capture["t"]
                            self._online=True; self._error=None; self._retry_at=None
                        delay=self._initial
                        if self._stop.wait(max(0,self._period-(self._clock()-began))): break
                except Exception as exc:
                    self._offline(exc,delay)
                    print(json.dumps({"viewer":"offline","reason":str(exc)[:240],"retry_in_s":delay}),flush=True)
                finally:
                    if client is not None:
                        try: client.close()
                        except Exception as exc: print(json.dumps({"viewer":"cleanup_error","reason":str(exc)[:240]}),flush=True)
                if self._stop.is_set() or self._stop.wait(delay): break
                delay=min(delay*2,self._maximum)
        finally: self._offline("Viewer stopped")
    def open(self):
        if self._thread is not None: raise RuntimeError("Viewer capture thread already started")
        self._thread=threading.Thread(target=self._run,name="kitchen-view-capture",daemon=False)
        self._thread.start()
    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive(): raise RuntimeError("Viewer capture thread exceeded its shutdown I/O budget")
            self._thread=None
        self._offline("Viewer stopped")


class KitchenStreamServer(StreamServer):
    """Offline card replaces unavailable imagery, including already-open MJPEG tabs."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self._offline_cards={}
    def _offline_jpeg(self,name):
        self._rig.get(name)
        if name not in self._offline_cards:
            # OpenCV uses BGR: match the paper UI even during camera recovery.
            img=np.full((360,640,3),(242,246,247),dtype=np.uint8)
            for text,y,size,color in [("OpenClaw Demo / "+name.capitalize(),65,.7,(34,39,36)),
                    ("Camera offline",153,.95,(41,90,121)),
                    ("Waiting for Isaac Sim. Reconnecting automatically.",200,.53,(94,105,101)),
                    ("The last capture is not shown as live video.",270,.43,(94,105,101))]:
                cv2.putText(img,text,(36,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
            ok,jpeg=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,90])
            if not ok: raise RuntimeError("Failed to encode offline card")
            self._offline_cards[name]=jpeg.tobytes()
        return self._offline_cards[name]
    def annotated_jpeg(self,name):
        # The demo UI supplies named views and freshness state. Show the full
        # untouched camera image here; debug HUD text would cover the scene.
        frame=self._rig.get(name).latest()
        if frame is None: return self._offline_jpeg(name)
        key=(frame.frame_id, frame.t, "demo-camera")
        with self._jpeg_lock:
            cached=self._jpeg_cache.get(name)
            if cached is not None and cached[0]==key: return cached[1]
        ok,jpeg=cv2.imencode('.jpg',frame.rgb,[cv2.IMWRITE_JPEG_QUALITY,self._quality])
        if not ok: return self._offline_jpeg(name)
        data=jpeg.tobytes()
        with self._jpeg_lock: self._jpeg_cache[name]=(key,data)
        return data
    def depth_jpeg(self,name): return super().depth_jpeg(name) or self._offline_jpeg(name)


def serve(stop_event,*,client_factory=None,server_factory=KitchenStreamServer):
    rig=RecoveringViewRig(client_factory=client_factory,stop_event=stop_event)
    server=server_factory(rig,state_fn=rig.state,host="127.0.0.1",port=8091,fps=8,quality=90)
    try:
        server.start()  # Listener survives simulator downtime; no warmup gate.
        rig.open()
        print(json.dumps({"url":URL,"camera":"kitchen","read_only":True}),flush=True)
        stop_event.wait()
    finally:
        try: rig.close()
        finally: server.stop()


def main():
    stop=threading.Event()
    def request_stop(*_): stop.set()
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    serve(stop)


if __name__=="__main__": main()
