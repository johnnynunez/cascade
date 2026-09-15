"""Security boundaries for the optional camera relay configuration."""

import unittest

from camera_stream_profile import StreamProfile


class StreamProfileSecurityTests(unittest.TestCase):
    def test_rejects_non_tailnet_streaming_addresses(self):
        for address in ("8.8.8.8", "0.0.0.0", "127.0.0.1", "192.168.1.2", "::1", "demo.example.com"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                StreamProfile(address, "https://demo.example.com")

    def test_rejects_credentials_and_insecure_visitor_origins(self):
        for origin in ("http://demo.example.com", "https://user:secret@demo.example.com", "https://demo.example.com/token"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                StreamProfile("100.100.100.100", origin)

    def test_rejects_port_collisions(self):
        for options in (
            {"rtsp_ports": (8554, 8554, 8556)},
            {"http_port": 8554},
            {"media_port": 49100},
            {"rtsp_ports": (8554, 8555, 8611)},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                StreamProfile("100.100.100.100", "https://demo.example.com", **options)

    def test_relay_only_exposes_tailnet_media_and_loopback_read_requests(self):
        config = StreamProfile("100.100.100.100", "https://demo.example.com").mediamtx_config()
        self.assertEqual(config["webrtcAddress"], "127.0.0.1:8889")
        self.assertEqual(config["webrtcLocalUDPAddress"], "100.100.100.100:8189")
        self.assertFalse(config["webrtcIPsFromInterfaces"])
        self.assertEqual(config["webrtcICEServers2"], [])
        for protocol in ("rtsp", "rtmp", "hls", "srt", "moq", "api", "metrics", "pprof", "playback"):
            self.assertFalse(config[protocol])
        users = config["authInternalUsers"]
        self.assertEqual(len(users), 1)
        self.assertEqual(set(users[0]["ips"]), {"127.0.0.1", "::1"})
        self.assertEqual({permission["action"] for permission in users[0]["permissions"]}, {"read"})
        self.assertEqual({permission["path"] for permission in users[0]["permissions"]}, {"kitchen", "worktop", "side"})

    def test_exact_tailnet_http_origin_preserves_private_relay(self):
        origin = "http://100.100.100.100:8092"
        config = StreamProfile("100.100.100.100", origin).mediamtx_config()
        self.assertEqual(config["webrtcAllowOrigins"], [origin])
        self.assertEqual(config["webrtcAddress"], "127.0.0.1:8889")
        self.assertEqual(config["webrtcLocalUDPAddress"], "100.100.100.100:8189")
        self.assertEqual(config["authInternalUsers"][0]["ips"], ["127.0.0.1", "::1"])
        self.assertFalse(config["webrtcIPsFromInterfaces"])

    def test_other_http_origins_are_rejected(self):
        for host in ("100.100.100.101", "8.8.8.8", "127.0.0.1", "demo.example.com",
                     "demo.tailnet.ts.net", "100.100.100.100.example.com"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                StreamProfile("100.100.100.100", f"http://{host}:8092")

    def test_malformed_or_credentialed_origins_are_rejected(self):
        origins = [
            "http://user:secret@100.100.100.100:8092",
            "http://@100.100.100.100:8092",
            "https://@demo.example.com",
            "http://100.100.100.100:8092/",
            "http://100.100.100.100:8092?",
            "https://demo.example.com#",
        ]
        for codepoint in (0, 9, 10, 13, 31, 127, 133):
            origins.append(chr(codepoint) + "http://100.100.100.100:8092")
            origins.append("https://demo" + chr(codepoint) + ".example.com")
        for origin in origins:
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                StreamProfile("100.100.100.100", origin)

    def test_invalid_origin_ports_are_rejected(self):
        for base in ("http://100.100.100.100", "https://demo.example.com"):
            for port in ("0", "65536", "-1", "invalid", ""):
                with self.subTest(base=base, port=port), self.assertRaises(ValueError):
                    StreamProfile("100.100.100.100", f"{base}:{port}")

    def test_https_and_valid_port_boundaries_are_accepted(self):
        for origin in ("https://demo.example.com", "https://demo.example.com:443",
                       "https://demo.example.com:65535", "http://100.100.100.100:1"):
            with self.subTest(origin=origin):
                config = StreamProfile("100.100.100.100", origin).mediamtx_config()
                self.assertEqual(config["webrtcAllowOrigins"], [origin])

    def test_existing_runtime_ports_are_reserved(self):
        for port in (8041, 8888, 18790, 18791, 47999, 48000, 49101, 49102):
            with self.subTest(port=port), self.assertRaises(ValueError):
                StreamProfile("100.100.100.100", "http://100.100.100.100:8092", media_port=port)


if __name__ == "__main__":
    unittest.main()
