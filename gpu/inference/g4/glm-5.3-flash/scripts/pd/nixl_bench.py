#!/usr/bin/env python3
"""Standalone NIXL READ benchmark between two nodes, mimicking what vLLM's NixlConnector does
(the decode side READs the prefill side's KV blocks). Isolates transfer speed from vLLM.

target (prefill side):   python3 nixl_bench.py target --port 18515
initiator (decode side): python3 nixl_bench.py init --host <target_ip> --port 18515 \
                             --total-mb 80 --descs 300 --mem cuda --iters 10 --streams 1

--descs: how many descriptors one transfer is split into (vLLM sent ~300 per request per rank).
--streams: concurrent transfers in flight (several requests / ranks at once).
UCX knobs come from the environment (UCX_TLS, UCX_TCP_*, ...).
"""
import argparse
import base64
import json
import socket
import time

import torch
from nixl._api import nixl_agent, nixl_agent_config


def alloc(total_bytes, mem):
    dev = "cuda" if mem == "cuda" else "cpu"
    t = torch.ones(total_bytes, dtype=torch.uint8, device=dev)
    if mem == "cpu" and torch.cuda.is_available():
        t = t.pin_memory()
    return t


def descs_of(t, n, mem):
    step = t.numel() // n
    base = t.data_ptr()
    dev = t.device.index or 0 if mem == "cuda" else 0
    return [(base + i * step, step, dev) for i in range(n)]


def send_msg(sock, obj):
    b = json.dumps(obj).encode()
    sock.sendall(len(b).to_bytes(8, "big") + b)


def recv_msg(sock):
    n = int.from_bytes(sock.recv(8, socket.MSG_WAITALL), "big")
    buf = b""
    while len(buf) < n:
        buf += sock.recv(n - len(buf))
    return json.loads(buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["target", "init"])
    ap.add_argument("--host", default="")
    ap.add_argument("--port", type=int, default=18515)
    ap.add_argument("--total-mb", type=float, default=80)
    ap.add_argument("--descs", type=int, default=300)
    ap.add_argument("--mem", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--streams", type=int, default=1)
    a = ap.parse_args()

    total = int(a.total_mb * 2**20)
    mt = "VRAM" if a.mem == "cuda" else "DRAM"
    agent = nixl_agent(a.role, nixl_agent_config(backends=["UCX"]))

    if a.role == "target":
        bufs = [alloc(total, a.mem) for _ in range(16)]  # enough for --streams up to 16
        for b in bufs:
            agent.register_memory(agent.get_reg_descs([(b.data_ptr(), b.numel(), b.device.index or 0 if a.mem == "cuda" else 0, "")], mt))
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", a.port)); srv.listen(1)
        print("target ready", flush=True)
        while True:
            c, _ = srv.accept()
            req = recv_msg(c)
            send_msg(c, {"meta": base64.b64encode(agent.get_agent_metadata()).decode(),
                         "bufs": [[b.data_ptr(), b.numel(), b.device.index or 0 if a.mem == "cuda" else 0] for b in bufs]})
            recv_msg(c)  # wait for "done"
            c.close()
            if req.get("last"):
                break
        return

    # initiator
    c = socket.create_connection((a.host, a.port))
    send_msg(c, {"last": True})
    info = recv_msg(c)
    remote = agent.add_remote_agent(base64.b64decode(info["meta"]))
    locs = [alloc(total, a.mem) for _ in range(a.streams)]
    for b in locs:
        agent.register_memory(agent.get_reg_descs([(b.data_ptr(), b.numel(), b.device.index or 0 if a.mem == "cuda" else 0, "")], mt))
    handles = []
    for s in range(a.streams):
        rb = info["bufs"][s]
        step = total // a.descs
        rd = [(rb[0] + i * step, step, rb[2]) for i in range(a.descs)]
        ld = descs_of(locs[s], a.descs, a.mem)
        handles.append(agent.initialize_xfer("READ", agent.get_xfer_descs(ld, mt), agent.get_xfer_descs(rd, mt), remote))
    # warmup (also establishes connections)
    for h in handles:
        agent.transfer(h)
    for h in handles:
        while agent.check_xfer_state(h) == "PROC":
            pass
    times = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        for h in handles:
            st = agent.transfer(h)
            assert st != "ERR"
        for h in handles:
            while (st := agent.check_xfer_state(h)) == "PROC":
                pass
            assert st == "DONE", st
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2]
    gbps = total * a.streams * 8 / med / 1e9
    print(json.dumps({"mem": a.mem, "total_mb": a.total_mb, "descs": a.descs, "streams": a.streams,
                      "median_ms": round(med * 1000, 1), "per_xfer_MBps": round(total / 2**20 / med, 0),
                      "aggregate_Gbps": round(gbps, 2)}), flush=True)
    send_msg(c, {"done": True})


if __name__ == "__main__":
    main()
