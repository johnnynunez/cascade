"""Build the optional video configuration from the admitted Brev profile."""

from __future__ import annotations

import json

from .camera_stream_profile import StreamProfile

RELAY_IMAGE = 'bluenviron/mediamtx@sha256:095da39dd94defa496d592e8b7a968bfd171491ea325d5176c6d32929ba4f1f9'


def enabled(profile):
    value = profile.get('streaming', {}).get('enabled', False)
    if type(value) is not bool:
        raise ValueError('The streaming profile needs an explicit boolean enabled value')
    return value


def configuration(profile, tailnet_ipv4):
    if not enabled(profile):
        return {}
    if profile.get('architecture') != 'x86_64':
        raise ValueError('The prepared camera relay image requires the Brev x86_64 profile')
    video = StreamProfile(tailnet_ipv4, f'http://{tailnet_ipv4}:8092')
    values = {'schema': 1, 'enabled': True, 'tailnet_ipv4': video.tailnet_ipv4,
              'visitor_origin': video.visitor_origin}
    return {'camera-video.json': json.dumps(values, indent=2) + '\n',
            'mediamtx.yml': json.dumps(video.mediamtx_config(), indent=2) + '\n'}
