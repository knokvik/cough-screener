"""
Diagnostic Audio Stream Tester (Step 2 Verification Tool)
Runs a standalone WebSocket server on port 8000 at /ws/audio,
captures raw 16-bit PCM stream from the ESP32 for N seconds,
saves it to 'test_recording.wav', and validates audio quality metrics.

Usage:
  python tools/record_test.py --duration 10 --port 8000 --output test_recording.wav
"""

import argparse
import asyncio
import logging
import math
from pathlib import Path
import sys
import wave

import numpy as np
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("record_test")


async def record_stream(port: int, duration_sec: float, output_path: Path, sample_rate: int = 16000):
    logger.info("==================================================================")
    logger.info(f"  Step 2 Audio Stream Verification Tool")
    logger.info(f"  Listening on: ws://0.0.0.0:{port}/ws/audio")
    logger.info(f"  Recording Duration: {duration_sec} seconds")
    logger.info(f"  Output WAV File: {output_path}")
    logger.info("==================================================================")
    logger.info("Waiting for ESP32 to connect over WiFi...")

    target_samples = int(duration_sec * sample_rate)
    target_bytes = target_samples * 2  # 16-bit = 2 bytes/sample
    recorded_bytes = bytearray()
    connection_event = asyncio.Event()

    async def audio_handler(websocket):
        path = getattr(websocket, "path", "/ws/audio")
        logger.info(f"[CONNECTED] ESP32 connected from {websocket.remote_address} on {path}!")
        connection_event.set()

        nonlocal recorded_bytes
        start_time = asyncio.get_event_loop().time()
        chunks_received = 0

        try:
            while len(recorded_bytes) < target_bytes:
                data = await websocket.recv()
                if isinstance(data, bytes):
                    recorded_bytes.extend(data)
                    chunks_received += 1
                    pct = min(100.0, (len(recorded_bytes) / target_bytes) * 100)
                    elapsed = asyncio.get_event_loop().time() - start_time
                    sys.stdout.write(f"\r[RECORDING] {pct:5.1f}% | {len(recorded_bytes) / 1024:6.1f} KB | Elapsed: {elapsed:4.1f}s")
                    sys.stdout.flush()

            sys.stdout.write("\n")
            logger.info("[SUCCESS] Recording target duration reached! Closing socket.")
            await websocket.close(1000, "Recording complete")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("[WARNING] ESP32 disconnected before target duration finished.")
        except Exception as e:
            logger.error(f"[ERROR] Exception during recording: {e}")

    server = await websockets.serve(audio_handler, "0.0.0.0", port)

    # Wait until connection occurs and completes
    try:
        while len(recorded_bytes) < target_bytes:
            await asyncio.sleep(0.1)
    finally:
        server.close()
        await server.wait_closed()

    if len(recorded_bytes) == 0:
        logger.error("No audio bytes received. Check ESP32 WiFi connection and IP address.")
        return

    # Trim to exact target length
    final_pcm = bytes(recorded_bytes[:target_bytes])

    # Save to WAV file
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)      # Mono
        wf.setsampwidth(2)      # 16-bit (2 bytes)
        wf.setframerate(sample_rate)
        wf.writeframes(final_pcm)

    logger.info(f"Saved {len(final_pcm)} bytes to {output_path}")

    # Analyze audio quality
    samples = np.frombuffer(final_pcm, dtype=np.int16)
    peak = int(np.max(np.abs(samples)))
    mean_sq = np.mean(samples.astype(np.float64) ** 2)
    rms = math.sqrt(mean_sq)
    zero_count = np.count_nonzero(samples == 0)
    zero_pct = (zero_count / len(samples)) * 100

    print("\n" + "=" * 60)
    print("                 AUDIO QUALITY REPORT")
    print("=" * 60)
    print(f"File Duration : {len(samples) / sample_rate:.2f} seconds")
    print(f"Sample Rate   : {sample_rate} Hz (16-bit Mono)")
    print(f"Peak Value    : {peak} / 32767 ({peak / 32767 * 100:.1f}% FS)")
    print(f"RMS Level     : {rms:.1f} ({20 * math.log10(max(1e-5, rms / 32768)):.1f} dBFS)")
    print(f"Zero Samples  : {zero_pct:.2f}%")
    print("-" * 60)

    if peak == 0:
        print("❌ STATUS: ALL ZEROS DETECTED!")
        print("   Troubleshooting: Check INMP441 wiring (SCK, WS, SD, L/R=GND, 3.3V).")
    elif peak > 32000:
        print("⚠️ STATUS: CLIPPING WARNING!")
        print("   Audio was loud or saturated. Consider increasing BIT_SHIFT in firmware.")
    elif rms < 50:
        print("⚠️ STATUS: VERY LOW SIGNAL!")
        print("   Microphone is very quiet. Bring mic closer or decrease BIT_SHIFT to 12.")
    else:
        print("✅ STATUS: CLEAN AUDIO DETECTED!")
        print(f"   Play the file to listen:  afplay {output_path} (macOS) or aplay {output_path} (Linux)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Record test audio stream from ESP32")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds (default: 10.0)")
    parser.add_argument("--output", type=str, default="test_recording.wav", help="Output WAV filename (default: test_recording.wav)")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate in Hz (default: 16000)")

    args = parser.parse_args()
    asyncio.run(record_stream(
        port=args.port,
        duration_sec=args.duration,
        output_path=Path(args.output),
        sample_rate=args.rate
    ))


if __name__ == "__main__":
    main()
