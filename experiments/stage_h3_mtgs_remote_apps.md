# Stage H3 MTGS Two-App Remote Driving Pilot

Date: 2026-07-26

## Question

Can the accepted MTGS simulated-driver loop be exposed as two separable LAN
applications while preserving the earlier TbV cockpit layout and keeping the
simulator authoritative?

This pilot tests the application boundary and a localhost transport loop. It
does not claim a real two-computer LAN latency result.

## Implemented Contract

The simulator application on the RTX host owns:

```text
AUTO or latest REMOTE HumanControl
→ 250 ms stale-control watchdog
→ SimpleVehicleModel
→ fail-closed +/-5 m route support
→ rigid three-camera MTGS render
→ calibrated 150-degree cylindrical panorama
→ truthful route/vehicle 3D inset
→ H.264 NVENC over SRT
```

The driver application owns keyboard shaping, AUTO/REMOTE/reset/estop
commands, SRT decoding, display, and telemetry presentation. Control and
telemetry use versioned JSON over WebSocket. A disconnected remote client
cannot directly retain control of world state.

The main view uses nearest-time `CAM_L0`, `CAM_F0`, and `CAM_R0` cameras; the
measured normalized time gaps from the centre camera are 0.00097/0/0.00099.
Each requested front pose is applied as one rigid delta to that calibrated
three-camera rig. The inset shows route geometry, the accepted corridor, and
the kinematic car; it is deliberately labelled as geometry rather than
reconstructed overhead RGB.

## GPU Visual/Performance Smoke

The accepted short v2 smoke used the released step-30,000 checkpoint on the
RTX 4090 D:

| Measurement | Result |
| --- | ---: |
| Output | 1280x544 at 20 FPS |
| Forward horizontal FOV | 150 degrees |
| Calibrated projection coverage | 100.0% |
| Three-camera render p50 / p95 | 22.16 / 27.80 ms |
| CPU panorama compose p50 / p95 | 20.32 / 20.97 ms |
| Peak CUDA reserved | 1.455 GiB |
| Scene time | fixed |

The initial 1600x668 attempt had 95.5% projection coverage, a visible lower
black edge, and about 33 ms panorama composition. Reducing only the remote
presentation raster to 1280x544 and changing the MTGS lower view angle from
-24 to -19 degrees removed the unobserved edge and brought the steady combined
render/compose time to about 47-48 ms. The 150-degree horizontal layout did not
change.

Short retained artifacts:

- `/home/yawei/stage3_external/artifacts/mtgs_remote_apps_20260726_smoke_v2.mp4`;
- `/home/yawei/stage3_external/artifacts/mtgs_remote_apps_20260726_smoke_v2.json`.

The final launcher-produced AUTO presentation contains 183 frames over
9.15 seconds. It reaches 11.994 m/s, completes actual +2.541/-2.671 m
vehicle-model excursions, retains 2.329 m minimum support margin, and stops at
the route endpoint with zero boundary hits. Manual inspection of the left
extremum, right recovery, and endpoint retained readable road/lane structure
and matching support-inset offsets. Artifacts:

- video:
  `/home/yawei/stage3_external/artifacts/mtgs_remote_apps_20260726/mtgs_remote_auto_demo.mp4`;
- video SHA-256:
  `95cd95e7338a21958c590570826ef8fdcaaec3bf4b21a12845f888b2dc7717d5`;
- JSON:
  `/home/yawei/stage3_external/artifacts/mtgs_remote_apps_20260726/mtgs_remote_auto_demo.json`.

## Two-Process Loopback

The final localhost run used the real MTGS model, WebSocket control on TCP
18766, and H.264 SRT video on UDP 19002. The simulator produced 120 frames.
The driver received all 120 telemetry messages and decoded 97 complete latest
video frames; it intentionally discards old decoded frames during initial GOP
acquisition. Both processes exited normally.

After adding explicit takeover assertions, a separate 60-frame regression on
TCP 18767 / UDP 19003 received 56 telemetry messages and 38 complete video
frames before shutdown, observed both `auto` and `remote`, and observed
non-zero applied steering (maximum absolute value 1.0). This directly verifies
that the headless client's W+A takeover command reached the authoritative
vehicle loop; the longer 120-frame run remains the transport-count evidence.

An earlier attempt is retained as failure evidence:

- TCP 8765 was already owned by an existing project service, so the new
  simulator correctly failed to bind rather than replacing it;
- a second attempt allowed the headless client's total timer to expire during
  model warm-up and used too small a 64 KB MPEG-TS probe, receiving telemetry
  but no complete video frame;
- the final client starts its smoke timer after first telemetry and uses a
  2 MB probe window.

## Client-Initiated Network Topology Revision

On 2026-07-27 the deployment constraint was clarified: the remote driving
computer can reach the Shidi simulator, but the Shidi computer cannot initiate
a connection back to the driver.

The control path already satisfied that constraint because the driver is the
WebSocket client. The SRT roles were reversed so the current default is:

```text
remote driving App
  ├─ WebSocket client/caller → Shidi TCP 18765 listener
  └─ SRT caller              → Shidi UDP 19001 listener

video payload on the established SRT connection flows Shidi → remote App
```

The native driver now exposes `--video-source`; the old `--video-listen`
spelling remains only as a compatibility alias. The Shidi launcher no longer
requires or uses the remote computer's address.

An exact-direction local transport smoke used an independent FFmpeg H.264
sender in `mode=listener` and receiver in `mode=caller`. The caller actively
connected and decoded all 60/60 frames of a 3-second, 128x72, 20 FPS synthetic
stream. Dependency-light tests also assert that the driver defaults to
`mode=caller` and the simulator defaults to `mode=listener`.

A subsequent real-checkpoint localhost two-process run used the revised
topology on TCP 18768 and UDP 19012. The simulator was the WebSocket and SRT
listener; the headless remote driving App initiated both connections. Over its
five-second acceptance interval the App received 69 complete video frames and
93 telemetry messages, observed both `auto` and `remote`, and observed maximum
absolute applied steering of 1.0 during W+A takeover. Both final client states
were connected. The simulator was then stopped normally with a keyboard
interrupt after the bounded client completed.

This passes SRT connection establishment, real MTGS video decoding, WebSocket
control/telemetry, and AUTO-to-REMOTE takeover in the required direction on
one host. It does not test two physical computers, firewall configuration,
wide-area reachability, or network latency.

## Fullscreen And Remote-Stream Hardening

On 2026-07-27 the first physical PowerShell-side observation reported two
problems: the native client was not fullscreen and the received video
frequently showed corrupt blocks.

The display issue was direct: the client used a fixed 0.8 display scale and
never requested fullscreen. The current client starts in true fullscreen,
resizes each complete frame to the largest aspect-preserving fit, keeps the
remaining area black, and uses Ctrl+Enter to toggle fullscreen.
`--no-fullscreen` retains an explicit windowed startup path. A subsequent
physical-client correction removed Escape as an application-exit shortcut:
Escape now only leaves fullscreen, while Ctrl+Q explicitly closes the App and
triggers its safety estop.

The stream had one proven configuration defect. The installed FFmpeg SRT
protocol help defines `latency`, `rcvlatency`, and `peerlatency` in
microseconds. The earlier URL used `latency=80`, which therefore requested
0.08 ms rather than the intended 80 ms and left effectively no remote
retransmission window. The revised transport uses:

- `latency=300000`, or 300 ms, on both peers;
- a 1,316-byte MPEG-TS-aligned SRT packet size;
- 8 Mbps rather than 12 Mbps H.264 CBR;
- NVENC p4/low-latency mode with spatial AQ;
- a 10-frame/0.5-second live GOP with forced IDR recovery;
- corrupt-packet discard in the client instead of the earlier no-buffer flag.

The latency and bitrate remain deployment knobs:
`MTGS_REMOTE_SRT_LATENCY_US=600000` plus
`MTGS_REMOTE_VIDEO_BITRATE=6M` is the documented conservative fallback.

An eight-second real-checkpoint localhost regression used the revised
listener/caller topology and exact new encoder/decoder options. The remote App
received 142 complete video frames and 151 telemetry messages, observed AUTO
and REMOTE, reached maximum absolute applied steering 1.0, and ended with both
video and control connected.

This validates option compatibility and the complete real-checkpoint software
path. It does not visually exercise the Windows Tk fullscreen window and does
not emulate real WAN jitter or packet loss, so elimination of field corruption
still requires one run on the actual two-computer path.

## PPT App-Control Evidence Clip

The reproducible offline presentation entry point is:

```bash
scripts/run_stage_h3_mtgs_remote.sh ppt-demo
```

It preserves the 1280x544 cockpit and adds a control-App panel below it,
producing a 1280x720, 20 FPS H.264 clip. The accepted final v2 contains 220
frames over 11.0 encoded seconds and presents:

```text
AUTO
→ REMOTE simulated operator through RemoteControlPacket
→ latched E-STOP
→ RESET
→ AUTO restart
```

All 100 scheduled remote packets were accepted. The panel derived its key
highlights from the actually applied control, yielding W/A/D on 100/45/36
frames. The estop applied full brake from frame 130 through 164, reducing
11.982 m/s to rest before reset. The run reached 2.671 m maximum absolute
lateral displacement, retained 2.329 m minimum support margin, and had zero
boundary hits. The three-camera render measured 25.66/28.05 ms p50/p95 and
panorama composition measured 20.36/21.66 ms p50/p95; peak reserved CUDA
memory was 1.455 GiB.

The final video decoded as 220/220 frames, has SHA-256
`82c55af63e0315aba5d4a081c6bbf12564008e99b7ebf714a247bae708a76222`,
and is retained with its machine-readable report and visual review images:

- `/home/yawei/stage3_external/artifacts/mtgs_app_control_ppt_20260727_v2/mtgs_app_control_ppt_demo.mp4`;
- `/home/yawei/stage3_external/artifacts/mtgs_app_control_ppt_20260727_v2/mtgs_app_control_ppt_demo.json`;
- `/home/yawei/stage3_external/artifacts/mtgs_app_control_ppt_20260727_v2/mtgs_app_control_ppt_preview.jpg`;
- `/home/yawei/stage3_external/artifacts/mtgs_app_control_ppt_20260727_v2/mtgs_app_control_ppt_contact_sheet.jpg`.

The operator is explicitly labelled as simulated. This clip exercises the
protocol objects, control authority, safety state, vehicle model, and real
checkpoint renderer in-process; it does not claim a physical keyboard,
WebSocket/SRT transport, two-computer LAN run, or network latency. The first
non-v2 render is retained separately as preliminary evidence because visual
inspection found one overlapping small-text row before the final layout fix.

## Verdict And Next Gate

Pass the two-application software boundary, three-camera cockpit, safety
watchdog, localhost WebSocket telemetry/control path, and localhost SRT video
path.

Do not claim real LAN latency, Internet security, dynamic traffic truth,
collision truth, a long route, or a tested desktop Tk window: the project host
was a display-less TTY and exercised the client's headless path. The immediate
application gate is one
five-minute two-computer LAN run with manual takeover, reset, estop, and
reconnect. The separate reconstruction next step remains the bounded
cross-traversal 3D-persistence experiment for the adjacent TbV tiles.
