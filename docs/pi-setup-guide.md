# Raspberry Pi Setup Guide: EEG Sleep Monitor

This guide walks through setting up a Raspberry Pi 4 Model B from a blank SD card to streaming live EEG data to the server.

## What You Need

**Hardware:**
- Raspberry Pi 4 Model B (2 GB+ RAM)
- microSD card (16 GB+)
- USB-C power supply (5V 3A)
- OpenBCI Cyton board with USB dongle (RFduino/CP210x)
- Ethernet cable or WiFi access
- Another computer for initial SD card flashing and SSH

**Accounts / Info:**
- Your server URL (e.g., `wss://your-app.railway.app`)
- A device ID registered in the dashboard (e.g., `pi-lab-01`)

---

## Step 1: Flash the SD Card

1. Download **Raspberry Pi Imager** from https://www.raspberrypi.com/software/ on your computer.
2. Insert the microSD card into your computer.
3. Open Raspberry Pi Imager and select:
   - **Device:** Raspberry Pi 4
   - **OS:** Raspberry Pi OS (64-bit) — under "Raspberry Pi OS (other)", pick the **Lite** version (no desktop needed)
   - **Storage:** Your microSD card
4. Click the **gear icon** (or "Edit Settings") before writing to pre-configure:
   - **Hostname:** e.g., `eeg-pi-01`
   - **Enable SSH:** Yes, with password authentication
   - **Username / Password:** Pick something (e.g., `pi` / your password)
   - **WiFi:** Enter your network name (SSID) and password
   - **Locale:** Set your timezone
5. Click **Write** and wait for it to finish.
6. Eject the SD card and insert it into the Pi.

---

## Step 2: Boot and Connect via SSH

1. Plug in ethernet (if available) or rely on the WiFi configured in Step 1.
2. Plug in the USB-C power supply. The Pi will boot automatically.
3. Wait ~60 seconds for the first boot to complete.
4. Find the Pi's IP address. Try one of these from your computer:
   ```bash
   # If you set hostname to eeg-pi-01:
   ping eeg-pi-01.local

   # Or check your router's admin page for connected devices

   # Or scan your network (replace with your subnet):
   nmap -sn 192.168.1.0/24
   ```
5. SSH into the Pi:
   ```bash
   ssh pi@eeg-pi-01.local
   # Or: ssh pi@192.168.1.XXX
   ```
   Enter the password you set in Step 1.

---

## Step 3: Update the System

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

This may take a few minutes. Reboot if prompted:
```bash
sudo reboot
```
Then SSH back in.

---

## Step 4: Install System Dependencies

```bash
sudo apt-get install -y \
    python3-venv python3-dev \
    build-essential cmake \
    libopenblas-dev libatlas-base-dev \
    git usbutils
```

These provide:
- Python virtual environment support
- Build tools for compiling native extensions (numpy, scipy)
- Optimized math libraries (OpenBLAS) for signal processing
- Git for cloning the project
- USB utilities for verifying the Cyton dongle

---

## Step 5: Clone the Project

```bash
cd ~
git clone https://github.com/myles-fb/EEG-Sleep-Monitor.git capstone
cd capstone
```

---

## Step 6: Create a Virtual Environment and Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

Install only the packages the Pi needs (not the full server/dashboard stack):

```bash
pip install \
    "brainflow>=5.11.0" \
    "numpy>=1.21.0" \
    "scipy>=1.7.0" \
    "scikit-learn>=1.0.0" \
    "websockets>=12.0" \
    "requests>=2.31.0"
```

This will take several minutes on the Pi as some packages compile from source.

Verify the install:
```bash
python -c "from brainflow.board_shim import BoardShim; print('BrainFlow OK')"
python -c "import scipy.signal; print('SciPy OK')"
python -c "from sklearn.linear_model import lasso_path; print('sklearn OK')"
python -c "import websockets; print('WebSockets OK')"
```

All four should print "OK". If any fail, see [Troubleshooting](#troubleshooting) below.

---

## Step 7: Set Up Serial Port Permissions

The Cyton board's USB dongle communicates over a serial port. Your user needs permission to access it.

```bash
sudo usermod -a -G dialout $USER
```

**You must log out and log back in** for this to take effect:
```bash
exit
# Then SSH back in
ssh pi@eeg-pi-01.local
```

Verify:
```bash
groups
# Should include "dialout" in the list
```

---

## Step 8: Connect the Cyton Board

1. Plug the Cyton USB dongle into one of the Pi's USB ports.
2. Turn on the Cyton board (slide the power switch).
3. Check that the Pi sees the dongle:
   ```bash
   ls /dev/ttyUSB*
   ```
   You should see `/dev/ttyUSB0`. If you see `/dev/ttyACM0` instead, use that in later steps.

   If nothing shows up:
   ```bash
   # Check kernel messages for USB detection
   dmesg | tail -20

   # List USB devices
   lsusb
   ```

---

## Step 9: Register the Device on the Dashboard

Before the Pi can stream, the device must be registered in the dashboard.

1. Open the physician dashboard in your browser (e.g., `https://your-app.railway.app`).
2. Go to the **Devices** page.
3. Click **Register New Device**.
4. Enter:
   - **Name:** A friendly name (e.g., "Pi Lab 01")
   - **Device Key:** A unique identifier (e.g., `pi-lab-01`) — this is the `--device-id` you'll use on the Pi
5. After registering, assign a **patient** to the device.
6. Start a **live study** for that patient from the New Study page (select "Live / Pi Streaming" as the source).

---

## Step 10: Test with Synthetic Data (No Hardware Needed)

Before connecting real hardware, verify the full pipeline works:

```bash
cd ~/capstone/src
source ~/capstone/.venv/bin/activate

python -m pi.pi_main \
    --synthetic \
    --server wss://your-app.railway.app/ws/pi/pi-lab-01 \
    --device-id pi-lab-01 \
    --verbose
```

Replace `wss://your-app.railway.app` with your actual server URL.

You should see:
```
INFO - Connecting to wss://your-app.railway.app/ws/pi/pi-lab-01
INFO - WebSocket connected
INFO - Registered with server
INFO - Starting acquisition (synthetic board)...
INFO - Processing window 0-300s...
```

Check the dashboard — you should see the device status change to "connected" on the Devices page, and data should start appearing on the Dashboard after the first processing window (5 minutes by default).

Press `Ctrl+C` to stop.

---

## Step 11: Stream Live EEG Data

Once the synthetic test works, switch to the real Cyton board:

```bash
cd ~/capstone/src
source ~/capstone/.venv/bin/activate

python -m pi.pi_main \
    --serial-port /dev/ttyUSB0 \
    --server wss://your-app.railway.app/ws/pi/pi-lab-01 \
    --device-id pi-lab-01
```

The Pi will:
1. Connect to the Cyton board at 250 Hz
2. Connect to the server via WebSocket
3. Every 5 minutes, process the buffered EEG data and send features to the server
4. Auto-reconnect if the server connection drops

---

## Step 12: Run as a Background Service (Optional)

To keep the streamer running after you close SSH (and auto-start on boot):

1. Copy and edit the service file:
   ```bash
   sudo cp ~/capstone/pi/eeg_streamer.service /etc/systemd/system/
   sudo nano /etc/systemd/system/eeg_streamer.service
   ```

2. Update these lines to match your setup:
   ```ini
   User=pi                          # Your Pi username
   WorkingDirectory=/home/pi/capstone/src
   ExecStart=/home/pi/capstone/.venv/bin/python -m pi.pi_main \
       --serial-port /dev/ttyUSB0 \
       --server wss://your-app.railway.app/ws/pi/pi-lab-01 \
       --device-id pi-lab-01 \
       --config /home/pi/capstone/pi_config.json
   ```

3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable eeg_streamer.service
   sudo systemctl start eeg_streamer.service
   ```

4. Check status and logs:
   ```bash
   # Service status
   sudo systemctl status eeg_streamer.service

   # Live logs
   journalctl -u eeg_streamer.service -f

   # Stop the service
   sudo systemctl stop eeg_streamer.service
   ```

The service will automatically restart if it crashes (after a 10-second delay) and start on boot.

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `sudo systemctl start eeg_streamer` | Start streaming |
| `sudo systemctl stop eeg_streamer` | Stop streaming |
| `sudo systemctl status eeg_streamer` | Check status |
| `journalctl -u eeg_streamer -f` | View live logs |
| `ls /dev/ttyUSB*` | Check if Cyton dongle is detected |
| `groups` | Verify dialout group membership |

---

## Troubleshooting

### "Permission denied: /dev/ttyUSB0"
You need serial port access. Run `sudo usermod -a -G dialout $USER`, then **log out and back in**.

### "No module named brainflow"
Make sure you activated the virtual environment: `source ~/capstone/.venv/bin/activate`

### Cyton dongle not detected (no `/dev/ttyUSB*`)
- Check the dongle is plugged in and the LED is on.
- Try a different USB port.
- Run `dmesg | tail -20` to see if the kernel recognized the device.
- Run `lsusb` to list connected USB devices.

### "Connection refused" or WebSocket won't connect
- Verify the server is running: open the dashboard URL in a browser.
- Check the device is registered on the Devices page with the correct device key.
- Make sure you're using `wss://` (not `ws://`) for HTTPS-deployed servers.
- Check the Pi has internet access: `ping google.com`

### scipy or numpy fails to install
On Raspberry Pi OS Bookworm, try installing the system packages as a fallback:
```bash
sudo apt-get install -y python3-numpy python3-scipy
```
Then recreate the venv with system packages:
```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install brainflow scikit-learn websockets requests
```

### Processing takes too long
The MOs detection pipeline is the most CPU-intensive step. If processing a 5-minute window takes more than 2-3 minutes, you can reduce the load by:
- Using fewer active channels in the patient config
- Reducing the algorithm window (e.g., to 120 seconds)
- Disabling MOs detection and using bandpower only

### Pi loses WiFi connection
For reliable overnight monitoring, use an ethernet cable instead of WiFi. The Pi client will auto-reconnect to the server when network access is restored.
