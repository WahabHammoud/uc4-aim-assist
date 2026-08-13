# Install ViGEm Bus Driver (Required for Stick Assist)

ViGEm Bus Driver lets the aim assist system create a virtual DualSense / Xbox controller that Chiaki sees. Without it, the right stick will not move and Milestone 2 will not work.

---

## Step 1 — Download ViGEm Bus Driver

1. Open this URL in a browser:
   ```
   https://github.com/nefarius/ViGEmBus/releases/latest
   ```
2. Under **Assets**, download `ViGEmBus_Setup_<version>.exe`
   (example: `ViGEmBus_Setup_1.22.0.exe`)

---

## Step 2 — Install ViGEm Bus Driver

1. Run the downloaded `.exe` as Administrator
2. Click **Install** and follow the prompts
3. Restart Windows when prompted

---

## Step 3 — Install vgamepad Python package

Open a terminal in the `uc4_aim_assist` folder and run:

```
pip install vgamepad
```

---

## Step 4 — Verify the install

Run the session checklist to confirm everything is working:

```
python tools/session_checklist.py
```

You should see a green checkmark next to **ViGEm / vgamepad**.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Failed to create virtual gamepad` in logs | ViGEm Bus Driver not installed or wrong version. Redo Step 1–2. |
| `ModuleNotFoundError: vgamepad` | Run `pip install vgamepad` |
| Right stick moves but Chiaki ignores it | In Chiaki → Settings → Gamepad, select the virtual DS4 controller |
| ViGEm installer fails | Right-click installer → Run as Administrator |

---

## Chiaki Configuration

After ViGEm is installed, Chiaki must be told to use the virtual controller:

1. Open **Chiaki**
2. Go to **Settings → Gamepad**
3. Select **DS4 (virtual)** or the device named "vgamepad" in the list
4. Save and reconnect to PS5

The aim assist blends the physical DualSense input with AI corrections and sends the result through the virtual DS4. Chiaki sees only the virtual controller.
