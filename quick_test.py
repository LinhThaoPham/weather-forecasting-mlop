"""
Simple test to check if the Model Monitor button is clickable and functions properly.
"""
import time
import json

# Test 1: Check if endpoints respond
print("=" * 60)
print("TEST 1: Checking API endpoints")
print("=" * 60)

import subprocess
result = subprocess.run(
    ["curl", "-s", "-w", "\\nStatus: %{http_code}", "http://127.0.0.1:8001/model/registry"],
    capture_output=True,
    text=True
)
print("Registry endpoint response:")
print(result.stdout[:200] + "...")

result2 = subprocess.run(
    ["curl", "-s", "-w", "\\nStatus: %{http_code}", "http://127.0.0.1:8001/model/history"],
    capture_output=True,
    text=True
)
print("\nHistory endpoint response:")
print(result2.stdout)

# Test 2: Check if dashboard page loads
print("\n" + "=" * 60)
print("TEST 2: Checking dashboard page")
print("=" * 60)

result3 = subprocess.run(
    ["curl", "-s", "-w", "\\nStatus: %{http_code}", "http://127.0.0.1:5173/index.html"],
    capture_output=True,
    text=True
)
status_code = result3.stdout.split("Status: ")[1].strip() if "Status:" in result3.stdout else "?"
print(f"Dashboard HTTP status: {status_code}")

# Check for nav-monitor button in HTML
html = result3.stdout.split("Status:")[0]
if 'id="nav-monitor"' in html:
    print("✓ nav-monitor button found in HTML")
    # Find the onclick handler
    start = html.find('id="nav-monitor"')
    snippet = html[start:start+300]
    if "onclick=" in snippet:
        print("✓ onclick handler found")
        if "showModelMonitor" in snippet:
            print("✓ showModelMonitor function called in onclick")
    else:
        print("✗ onclick handler NOT found")
else:
    print("✗ nav-monitor button NOT found in HTML")

# Test 3: Check for script setup
print("\n" + "=" * 60)
print("TEST 3: Checking script.js")
print("=" * 60)

result4 = subprocess.run(
    ["curl", "-s", "http://127.0.0.1:5173/script.js"],
    capture_output=True,
    text=True
)
script = result4.stdout

if "function showModelMonitor()" in script:
    print("✓ showModelMonitor function defined")
else:
    print("✗ showModelMonitor function NOT defined")

if "window.showModelMonitor = showModelMonitor" in script:
    print("✓ showModelMonitor exported to window")
else:
    print("✗ showModelMonitor NOT exported to window")

if 'document.getElementById("nav-monitor")?.addEventListener("click"' in script:
    print("✓ Event listener for nav-monitor found")
else:
    print("✗ Event listener for nav-monitor NOT found")

if "loadMonitorData" in script:
    print("✓ loadMonitorData function found")
else:
    print("✗ loadMonitorData function NOT found")

# Test 4: Syntax check
print("\n" + "=" * 60)
print("TEST 4: Checking for common JS errors")
print("=" * 60)

if script.count("{") == script.count("}"):
    print("✓ Braces balanced")
else:
    print("✗ Braces NOT balanced - possible syntax error")

# Check for common issues
if 'async function loadMonitorData()' in script:
    print("✓ loadMonitorData is async")
else:
    print("✗ loadMonitorData might not be async")

if 'await Promise.all' in script and 'model/registry' in script:
    print("✓ loadMonitorData fetches /model/registry")
else:
    print("✗ loadMonitorData might not fetch /model/registry")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
