"""OpenRouter connection check, independent of ROS and MuJoCo rendering."""
import argparse
import numpy as np
from .cargo_vlm import DEFAULT_MODEL, DEFAULT_BASE_URL, api_key_for, vlm_decide

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--base-url', default=DEFAULT_BASE_URL)
    args = ap.parse_args()
    image = np.full((192, 192, 3), 220, dtype=np.uint8)
    image[48:144, 48:144] = [220, 25, 25]
    try:
        key = api_key_for(args.base_url)
        color, _ = vlm_decide(image, [x.strip() for x in args.model.split(',') if x.strip()], args.base_url, key)
    except Exception as exc:
        print(str(exc))
        raise SystemExit(1)
    print(f'API connection OK; synthetic red parcel classified as {color}')
    if color != 'red':
        print('API responded, but the synthetic image classification was incorrect.')
        raise SystemExit(2)

if __name__ == '__main__': main()
