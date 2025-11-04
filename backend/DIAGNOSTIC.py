import os
import json

print("🔍 DIAGNOSTIC CHECK FOR detection_logs/")
print("=" * 70)

# Check if directory exists
if not os.path.exists('detection_logs'):
    print("❌ detection_logs/ directory does NOT exist!")
    print("💡 Run your mobile app first to generate it")
    exit()

print("✅ detection_logs/ exists")

# Check realtime_detections.jsonl
jsonl_file = 'detection_logs/realtime_detections.jsonl'
if os.path.exists(jsonl_file):
    size = os.path.getsize(jsonl_file)
    print(f"✅ File exists: {jsonl_file}")
    print(f"📊 File size: {size} bytes")

    if size == 0:
        print("⚠️  FILE IS EMPTY! No data to analyze")
        print("🔴 SOLUTION: You need to RUN YOUR MOBILE APP FIRST!")
        print("\nSteps:")
        print("1. Terminal 1: ngrok http 5000")
        print("2. Terminal 2: python server.py")
        print("3. Terminal 3: npx expo start --tunnel")
        print("4. Use phone to run the app for 2-3 minutes")
        print("5. Then run this script again")
        exit()

    # Try to read the file
    try:
        with open(jsonl_file, 'r') as f:
            lines = f.readlines()

        print(f"📝 Total lines: {len(lines)}")

        # Check first few lines
        valid_lines = 0
        for i, line in enumerate(lines[:5]):
            try:
                data = json.loads(line.strip())
                valid_lines += 1
                print(f"  Line {i+1}: ✅ Valid JSON (frame_id={data.get('frame_id', '?')})")
            except:
                print(f"  Line {i+1}: ❌ Invalid JSON")

        print(f"\n✅ Valid JSON lines: {valid_lines}/{min(5, len(lines))}")

        # Count total valid lines
        total_valid = 0
        for line in lines:
            try:
                json.loads(line.strip())
                total_valid += 1
            except:
                pass

        print(f"📊 Total valid JSON lines: {total_valid}/{len(lines)}")

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        exit()
else:
    print(f"❌ File NOT found: {jsonl_file}")
    exit()

# Check frames directory
frames_dir = 'detection_logs/frames'
if os.path.exists(frames_dir):
    frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    print(f"✅ Frames directory: {frame_count} images")
else:
    print(f"⚠️  Frames directory missing: {frames_dir}")

# Check metrics directory
metrics_dir = 'detection_logs/metrics'
if os.path.exists(metrics_dir):
    metrics_count = len([f for f in os.listdir(metrics_dir) if f.endswith('.json')])
    print(f"✅ Metrics directory: {metrics_count} JSON files")
else:
    print(f"⚠️  Metrics directory missing: {metrics_dir}")

print("\n" + "=" * 70)
print("✅ DIAGNOSTIC COMPLETE")
print("=" * 70)