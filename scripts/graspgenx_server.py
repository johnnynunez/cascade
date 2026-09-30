#!/usr/bin/env python3
"""Run the upstream learned server with CUDA identity in its health reply."""
import argparse
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--default-gripper")
    args = parser.parse_args()
    import torch
    from graspgenx.serving.zmq_server import GraspGenXZMQServer

    if not torch.cuda.is_available():
        raise RuntimeError("GraspGen-X requires CUDA; CPU fallback is disabled")

    class CudaServer(GraspGenXZMQServer):
        def _dispatch(self, request):
            reply = super()._dispatch(request)
            if request.get("action") == "health":
                device = next(self._shared_model.parameters()).device
                if device.type != "cuda":
                    raise RuntimeError(f"GraspGen-X model is on {device}, expected CUDA")
                reply.update(learned=True, device=str(device),
                             gpu=torch.cuda.get_device_name(device),
                             capability=list(torch.cuda.get_device_capability(device)),
                             torch=torch.__version__, cuda=torch.version.cuda,
                             architecture=platform.machine())
            return reply

    CudaServer(config_path=args.config, assets_dir=args.assets_dir,
               host=args.host, port=args.port,
               default_gripper=args.default_gripper).serve_forever()


if __name__ == "__main__":
    main()
