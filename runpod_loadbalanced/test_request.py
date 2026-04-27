import base64
import json
import sys

import requests


def main():
    if len(sys.argv) != 4:
        print("Usage: python test_request.py ENDPOINT_ID RUNPOD_API_KEY AUDIO_FILE")
        raise SystemExit(1)

    endpoint_id = sys.argv[1]
    api_key = sys.argv[2]
    audio_path = sys.argv[3]

    with open(audio_path, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = requests.post(
        f"https://{endpoint_id}.api.runpod.ai/transcribe",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "audio_base64": audio_base64,
            "language": "en",
        },
        timeout=180,
    )

    print("Status:", response.status_code)
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
