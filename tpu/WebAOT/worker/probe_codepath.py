# trace 期打印：MoE 的分片规格 + shard_map 的 in/out specs + gmm 实际入参形状
import importlib
_seen=set()
def _p(*a):
    s=" ".join(str(x) for x in a)
    if s not in _seen: _seen.add(s); print("@@@PSP "+s, flush=True)

for mn in ("maxtext.layers.moe","src.maxtext.layers.moe"):
    try: M=importlib.import_module(mn)
    except Exception: continue
    C=getattr(M,"RoutedMoE",None)
    if C is None or getattr(C.sparse_matmul,"_psp",False): continue
    o=C.sparse_matmul
    def mk(o):
        def w(self, inputs, gate_logits, pre_bias_logits, w0, w1, wo, *a, **k):
            cfg=self.config
            _p("cfg shard_exp_on_fsdp=",cfg.shard_exp_on_fsdp,"num_experts=",cfg.num_experts,
               "ep_size=",self.get_expert_parallelism_size(),"quant=",bool(cfg.quantization))
            _p("mesh axes=",self.mesh.axis_names,"shape=",self.mesh.devices.shape)
            _p("wi_kernel_axes=",self.wi_kernel_axes,"-> pspec",self._logical_to_mesh_axes(self.wi_kernel_axes))
            _p("wo_kernel_axes=",self.wo_kernel_axes,"-> pspec",self._logical_to_mesh_axes(self.wo_kernel_axes))
            _p("activation_batch -> ",self._logical_to_mesh_axes(("activation_batch","activation_norm_length",None)))
            _p("GLOBAL x=",inputs.shape," w0=",w0.shape," w1=",w1.shape," wo=",wo.shape)
            return o(self, inputs, gate_logits, pre_bias_logits, w0, w1, wo, *a, **k)
        w._psp=True; return w
    C.sparse_matmul=mk(o); _p("patched",mn)

for mn in ("maxtext.kernels.megablox","maxtext.kernels.megablox.ops",
           "src.maxtext.kernels.megablox","src.maxtext.kernels.megablox.ops"):
    try: O=importlib.import_module(mn)
    except Exception: continue
    g=getattr(O,"gmm",None)
    if g is None or getattr(g,"_psp",False): continue
    def mk2(o):
        def w(lhs,rhs,group_sizes,*a,**k):
            sh=lambda t: tuple(getattr(t,"shape",()) )
            q=getattr(rhs,"qvalue",None)
            _p("KERNEL lhs=",sh(lhs)," rhs=",sh(rhs), (" qv="+str(sh(q)) if q is not None else ""),
               " gs=",sh(group_sizes)," wga=",k.get("weight_gather_axes"),
               " tokamax=",k.get("use_tokamax_backend"))
            return o(lhs,rhs,group_sizes,*a,**k)
        w._psp=True; return w
    O.gmm=mk2(g)
