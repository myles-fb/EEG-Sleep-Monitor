# Testing Pi ↔ Railway Communication (No Cyton Board Needed)

This procedure verifies the full data pipeline from a simulated EEG source through the Railway-hosted server to the dashboard, using BrainFlow's synthetic board instead of real hardware.

## Prerequisites

- Your Railway app is deployed and running
- You have the Railway public domain (for the dashboard) and TCP proxy address (for WebSocket)
- Python 3.9+ with the project dependencies installed (on your laptop or a Pi)

## Step 1: Get your Railway URLs

From the Railway dashboard, find:
- **Dashboard URL**: something like `https://your-app.up.railway.app` (under your service → Settings → Networking → Public Domain)
- **WebSocket URL**: the TCP proxy address like `your-app.proxy.rlwy.net:PORT` (under Networking → TCP Proxy for port 8765)

Open the dashboard URL in a browser and confirm it loads.

## Step 2: Register a test device

1. Open `https://your-app.up.railway.app` in your browser
2. Go to the **Devices** page
3. Click **Register New Device**
4. Enter:
   - **Name**: `Synthetic Test`
   - **Device Key**: `pi-test`

## Step 3: Create a patient and assign to device

1. Go to the **Home** page, create a patient (e.g., "Test Patient")
2. Go to the **Devices** page, assign **Test Patient** to the **Synthetic Test** device
3. Go to **New Study**, select **Test Patient**, choose **Live / Pi Streaming**, and start the study

## Step 4: Run the synthetic streamer

On your laptop (or Pi), from the project root:

```bash
cd src
source ../.venv/bin/activate

python -m pi.pi_main \
    --synthetic \
    --server wss://your-app.proxy.rlwy.net:PORT/ws/pi/pi-test \
    --device-id pi-test \
    --verbose
```

Replace `your-app.proxy.rlwy.net:PORT` with your actual TCP proxy address.

## Step 5: Verify connection

You should see logs like:
```
INFO - Connecting to wss://your-app.proxy.rlwy.net:PORT/ws/pi/pi-test
INFO - WebSocket connected
INFO - Registered with server
INFO - Starting acquisition (synthetic board)...
```

On the **Devices** page in the dashboard, the device status should change to **"connected"**.

## Step 6: Wait for first data window

The default algorithm window is **5 minutes**. After 5 minutes you should see:
```
INFO - Processing window 0-300s...
INFO - Features sent to server
```

To speed this up for testing, you can reduce the window. Create a config file:

```bash
cat > /tmp/pi_test_config.json << 'EOF'
{
    "algorithm_window": 30,
    "features": {
        "mo_q_score": true,
        "mo_count": true,
        "bandpower_delta": true,
        "bandpower_theta": true,
        "bandpower_alpha": true,
        "bandpower_beta": true
    },
    "active_channels": [0, 1],
    "notification_thresholds": {}
}
EOF
```

Then run with `--config /tmp/pi_test_config.json` to process every 30 seconds instead of 5 minutes:

```bash
python -m pi.pi_main \
    --synthetic \
    --server wss://your-app.proxy.rlwy.net:PORT/ws/pi/pi-test \
    --device-id pi-test \
    --config /tmp/pi_test_config.json \
    --verbose
```

## Step 7: Verify data on dashboard

1. Go to the **Dashboard** page
2. Select **Test Patient** and the active study
3. You should see Q-score, P-value, and MO count data appearing after the first window processes

## Step 8: Verify resilience

Test auto-reconnect by killing the streamer with `Ctrl+C`, waiting a few seconds, and restarting it. The device status should go to "offline" then back to "connected", and data should continue accumulating.

## What Success Looks Like

| Check | Expected |
|-------|----------|
| Streamer logs show "WebSocket connected" | Pi reached the Railway server |
| Devices page shows "connected" | Server recognized the device |
| Streamer logs show "Features sent to server" | Processing pipeline ran, data transmitted |
| Dashboard shows charts with data points | Server stored features in PostgreSQL, dashboard queries work |
| After Ctrl+C + restart, streaming resumes | Auto-reconnect works |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Connection refused" | Check the TCP proxy address and port — make sure you're using `wss://` not `ws://` |
| "Device not found" (4001 error) | Register the device on the Devices page first — the `--device-id` must match the device key exactly |
| Connected but no data on dashboard | Make sure you assigned a patient and started a live study before connecting |
| Timeout during processing | With `--synthetic` and 2 active channels, a 30s window should process in under 5 seconds. Check Railway logs for server-side errors |
